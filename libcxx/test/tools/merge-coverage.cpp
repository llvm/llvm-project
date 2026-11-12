#include <algorithm>
#include <flat_map>
#include <flat_set>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

struct SourceRange {
  uint64_t file;
  uint64_t expanded_file;
  uint64_t region_kind;
  uint64_t line_start;
  uint64_t column_start;
  uint64_t line_end;
  uint64_t column_end;

  friend auto operator<=>(const SourceRange&, const SourceRange&) = default;
};

struct ExecutionCounts {
  uint64_t true_count;
  uint64_t false_count;
};

struct FunctionInfo {
  std::vector<std::string> filenames;
  std::flat_map<SourceRange, uint64_t> regions;
  std::flat_map<SourceRange, ExecutionCounts> branches;
  uint64_t count;
};

struct [[nodiscard]] BalancedBraces {
  enum Kind {
    parens,
    square,
    brace,
  } kind;

  BalancedBraces(Kind k) : kind(k) {
    switch (kind) {
    case parens:
      std::cout << '(';
      break;
    case square:
      std::cout << '[';
      break;
    case brace:
      std::cout << '{';
      break;
    }
  }
  BalancedBraces(const BalancedBraces&)            = delete;
  BalancedBraces& operator=(const BalancedBraces&) = delete;
  ~BalancedBraces() {
    switch (kind) {
    case parens:
      std::cout << ')';
      break;
    case square:
      std::cout << ']';
      break;
    case brace:
      std::cout << '}';
      break;
    }
  }
};

int main(int, char** argv) {
  std::ios::sync_with_stdio(false);
  std::map<std::string, FunctionInfo> functions;

  using json = nlohmann::json;

  std::ifstream files(argv[1]);

  std::string fname;
  while (std::getline(files, fname)) {
    std::ifstream f(fname);
    if (f.peek() == EOF) // skip empty files
      continue;
    // TODO: Verify the the files have the expected layout
    json report = json::parse(f)["data"][0]["functions"];
    for (auto& file : report) {
      std::string function_name(file["name"]);

      if (function_name == "main")
        continue;

      FunctionInfo& func_info = functions[std::move(function_name)];
      auto& filenames         = file["filenames"];
      if (std::ranges::none_of(filenames, [](json j) {
            auto name = std::string(j);
            return name.contains("test-suite-install") || name.contains("libcxx/src") ||
                   name.contains("libcxxabi/src") || name.contains("libunwind/src");
          }))
        continue;
      if (func_info.filenames.empty()) {
        for (auto& filename : filenames)
          func_info.filenames.push_back(std::string(filename));
      } else {
        auto [lhs_res, rhs_res] =
            std::ranges::mismatch(filenames, func_info.filenames, {}, [](json& j) { return std::string(j); });
        if (lhs_res != filenames.end() && rhs_res != func_info.filenames.end()) {
          std::clog << "Rejected information for function '" << file["name"] << "' due to different file names\n";
          for (const auto& n : func_info.filenames)
            std::clog << n << '\n';
          std::clog << "vs.\n";
          for (const auto& n : filenames)
            std::clog << std::string(n) << '\n';
          continue;
        }
        std::transform(lhs_res, filenames.end(), std::back_inserter(func_info.filenames), [](json j) {
          return std::string(j);
        });
      }
      for (auto& region : file["regions"]) {
        if (uint64_t(region.at(5)) >= func_info.filenames.size() ||
            uint64_t(region.at(6)) >= func_info.filenames.size()) {
          std::clog << "Incoherent indices\n";
          continue;
        }
        func_info.regions[SourceRange{
            uint64_t(region.at(5)),
            uint64_t(region.at(6)),
            uint64_t(region.at(7)),
            uint64_t(region.at(0)),
            uint64_t(region.at(1)),
            uint64_t(region.at(2)),
            uint64_t(region.at(3))}] += uint64_t(region.at(4));
      }
      for (auto& branch : file["branches"]) {
        auto& [true_count, false_count] = func_info.branches[SourceRange{
            uint64_t(branch.at(6)),
            uint64_t(branch.at(7)),
            uint64_t(branch.at(8)),
            uint64_t(branch.at(0)),
            uint64_t(branch.at(1)),
            uint64_t(branch.at(2)),
            uint64_t(branch.at(3))}];
        true_count += uint64_t(branch.at(4));
        false_count += uint64_t(branch.at(5));
      }
      func_info.count += uint64_t(file["count"]);
    }
  }

  std::cout << R"json({"data":[{"functions":[)json";
  bool is_first = true;

  for (const auto& [name, info] : functions) {
    if (!is_first)
      std::cout << ',';
    is_first = false;

    BalancedBraces _(BalancedBraces::brace);

    std::cout << R"json("name":")json" << name << "\",";
    std::cout << R"json("count":)json" << info.count << ',';

    {
      std::cout << R"json("filenames":)json";
      BalancedBraces _(BalancedBraces::square);

      bool is_first_filename = true;
      for (const auto& filename : info.filenames) {
        if (!is_first_filename)
          std::cout << ',';
        is_first_filename = false;
        std::cout << '"' << filename << '"';
      }
    }
    std::cout << ',';

    {
      std::cout << R"json("branches":)json";
      BalancedBraces _(BalancedBraces::square);

      bool is_first_branch = true;
      for (const auto& [branch, branch_val] : info.branches) {
        if (!is_first_branch)
          std::cout << ',';
        is_first_branch = false;
        BalancedBraces _(BalancedBraces::square);
        std::cout << branch.line_start << ',' << branch.column_start << ',' << branch.line_end << ','
                  << branch.column_end << ',' << branch_val.true_count << ',' << branch_val.false_count << ','
                  << branch.file << ',' << branch.expanded_file << ',' << branch.region_kind;
      }
    }
    std::cout << ',';

    // TODO: Merge MCDC records
    std::cout << R"json("mcdc_records":[],)json";

    {
      std::cout << R"json("regions":)json";
      BalancedBraces _(BalancedBraces::square);

      bool is_first_region = true;
      for (const auto& [region, region_val] : info.regions) {
        if (!is_first_region)
          std::cout << ',';
        is_first_region = false;
        BalancedBraces _(BalancedBraces::square);
        std::cout << region.line_start << ',' << region.column_start << ',' << region.line_end << ','
                  << region.column_end << ',' << region_val << ',' << region.file << ',' << region.expanded_file << ','
                  << region.region_kind;
      }
    }
  }
  std::cout << "]}],";

  std::cout << R"json("type":"llvm.coverage.json.export","version":"3.1.0")json";

  std::cout << "}";
}
