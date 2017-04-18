#include <iostream>
#include <map>
#include <memory>
#include <sstream>

#include "../src/check.h"  // NOTE: check.h is for internal use only!
#include "../src/re.h"     // NOTE: re.h is for internal use only
#include "output_test.h"

// ========================================================================= //
// ------------------------------ Internals -------------------------------- //
// ========================================================================= //
namespace internal {
namespace {

using TestCaseList = std::vector<TestCase>;

// Use a vector because the order elements are added matters during iteration.
// std::map/unordered_map don't guarantee that.
// For example:
//  SetSubstitutions({{"%HelloWorld", "Hello"}, {"%Hello", "Hi"}});
//     Substitute("%HelloWorld") // Always expands to Hello.
using SubMap = std::vector<std::pair<std::string, std::string>>;

TestCaseList& GetTestCaseList(TestCaseID ID) {
  // Uses function-local statics to ensure initialization occurs
  // before first use.
  static TestCaseList lists[TC_NumID];
  return lists[ID];
}

SubMap& GetSubstitutions() {
  // Don't use 'dec_re' from header because it may not yet be initialized.
  static std::string safe_dec_re = "[0-9]*[.]?[0-9]+([eE][-+][0-9]+)?";
  static SubMap map = {
      {"%float", "[0-9]*[.]?[0-9]+([eE][-+][0-9]+)?"},
      {"%int", "[ ]*[0-9]+"},
      {" %s ", "[ ]+"},
      {"%time", "[ ]*[0-9]{1,5} ns"},
      {"%console_report", "[ ]*[0-9]{1,5} ns [ ]*[0-9]{1,5} ns [ ]*[0-9]+"},
      {"%console_us_report", "[ ]*[0-9] us [ ]*[0-9] us [ ]*[0-9]+"},
      {"%csv_report", "[0-9]+," + safe_dec_re + "," + safe_dec_re + ",ns,,,,,"},
      {"%csv_us_report", "[0-9]+," + safe_dec_re + "," + safe_dec_re + ",us,,,,,"},
      {"%csv_bytes_report",
       "[0-9]+," + safe_dec_re + "," + safe_dec_re + ",ns," + safe_dec_re + ",,,,"},
      {"%csv_items_report",
       "[0-9]+," + safe_dec_re + "," + safe_dec_re + ",ns,," + safe_dec_re + ",,,"},
      {"%csv_label_report_begin", "[0-9]+," + safe_dec_re + "," + safe_dec_re + ",ns,,,"},
      {"%csv_label_report_end", ",,"}};
  return map;
}

std::string PerformSubstitutions(std::string source) {
  SubMap const& subs = GetSubstitutions();
  using SizeT = std::string::size_type;
  for (auto const& KV : subs) {
    SizeT pos;
    SizeT next_start = 0;
    while ((pos = source.find(KV.first, next_start)) != std::string::npos) {
      next_start = pos + KV.second.size();
      source.replace(pos, KV.first.size(), KV.second);
    }
  }
  return source;
}

void CheckCase(std::stringstream& remaining_output, TestCase const& TC,
               TestCaseList const& not_checks) {
  std::string first_line;
  bool on_first = true;
  std::string line;
  while (remaining_output.eof() == false) {
    CHECK(remaining_output.good());
    std::getline(remaining_output, line);
    if (on_first) {
      first_line = line;
      on_first = false;
    }
    for (const auto& NC : not_checks) {
      CHECK(!NC.regex->Match(line))
          << "Unexpected match for line \"" << line << "\" for MR_Not regex \""
          << NC.regex_str << "\""
          << "\n    actual regex string \"" << TC.substituted_regex << "\""
          << "\n    started matching near: " << first_line;
    }
    if (TC.regex->Match(line)) return;
    CHECK(TC.match_rule != MR_Next)
        << "Expected line \"" << line << "\" to match regex \"" << TC.regex_str
        << "\""
        << "\n    actual regex string \"" << TC.substituted_regex << "\""
        << "\n    started matching near: " << first_line;
  }
  CHECK(remaining_output.eof() == false)
      << "End of output reached before match for regex \"" << TC.regex_str
      << "\" was found"
      << "\n    actual regex string \"" << TC.substituted_regex << "\""
      << "\n    started matching near: " << first_line;
}

void CheckCases(TestCaseList const& checks, std::stringstream& output) {
  std::vector<TestCase> not_checks;
  for (size_t i = 0; i < checks.size(); ++i) {
    const auto& TC = checks[i];
    if (TC.match_rule == MR_Not) {
      not_checks.push_back(TC);
      continue;
    }
    CheckCase(output, TC, not_checks);
    not_checks.clear();
  }
}

class TestReporter : public benchmark::BenchmarkReporter {
 public:
  TestReporter(std::vector<benchmark::BenchmarkReporter*> reps)
      : reporters_(reps) {}

  virtual bool ReportContext(const Context& context) {
    bool last_ret = false;
    bool first = true;
    for (auto rep : reporters_) {
      bool new_ret = rep->ReportContext(context);
      CHECK(first || new_ret == last_ret)
          << "Reports return different values for ReportContext";
      first = false;
      last_ret = new_ret;
    }
    (void)first;
    return last_ret;
  }

  void ReportRuns(const std::vector<Run>& report) {
    for (auto rep : reporters_) rep->ReportRuns(report);
  }
  void Finalize() {
    for (auto rep : reporters_) rep->Finalize();
  }

 private:
  std::vector<benchmark::BenchmarkReporter *> reporters_;
};
}
}  // end namespace internal

// ========================================================================= //
// -------------------------- Public API Definitions------------------------ //
// ========================================================================= //

TestCase::TestCase(std::string re, int rule)
    : regex_str(std::move(re)),
      match_rule(rule),
      substituted_regex(internal::PerformSubstitutions(regex_str)),
      regex(std::make_shared<benchmark::Regex>()) {
  std::string err_str;
  regex->Init(substituted_regex,& err_str);
  CHECK(err_str.empty()) << "Could not construct regex \"" << substituted_regex
                         << "\""
                         << "\n    originally \"" << regex_str << "\""
                         << "\n    got error: " << err_str;
}

int AddCases(TestCaseID ID, std::initializer_list<TestCase> il) {
  auto& L = internal::GetTestCaseList(ID);
  L.insert(L.end(), il);
  return 0;
}

int SetSubstitutions(
    std::initializer_list<std::pair<std::string, std::string>> il) {
  auto& subs = internal::GetSubstitutions();
  for (auto KV : il) {
    bool exists = false;
    KV.second = internal::PerformSubstitutions(KV.second);
    for (auto& EKV : subs) {
      if (EKV.first == KV.first) {
        EKV.second = std::move(KV.second);
        exists = true;
        break;
      }
    }
    if (!exists) subs.push_back(std::move(KV));
  }
  return 0;
}

void RunOutputTests(int argc, char* argv[]) {
  using internal::GetTestCaseList;
  benchmark::Initialize(&argc, argv);
  benchmark::ConsoleReporter CR(benchmark::ConsoleReporter::OO_None);
  benchmark::JSONReporter JR;
  benchmark::CSVReporter CSVR;
  struct ReporterTest {
    const char* name;
    std::vector<TestCase>& output_cases;
    std::vector<TestCase>& error_cases;
    benchmark::BenchmarkReporter& reporter;
    std::stringstream out_stream;
    std::stringstream err_stream;

    ReporterTest(const char* n, std::vector<TestCase>& out_tc,
                 std::vector<TestCase>& err_tc,
                 benchmark::BenchmarkReporter& br)
        : name(n), output_cases(out_tc), error_cases(err_tc), reporter(br) {
      reporter.SetOutputStream(&out_stream);
      reporter.SetErrorStream(&err_stream);
    }
  } TestCases[] = {
      {"ConsoleReporter", GetTestCaseList(TC_ConsoleOut),
       GetTestCaseList(TC_ConsoleErr), CR},
      {"JSONReporter", GetTestCaseList(TC_JSONOut), GetTestCaseList(TC_JSONErr),
       JR},
      {"CSVReporter", GetTestCaseList(TC_CSVOut), GetTestCaseList(TC_CSVErr),
       CSVR},
  };

  // Create the test reporter and run the benchmarks.
  std::cout << "Running benchmarks...\n";
  internal::TestReporter test_rep({&CR, &JR, &CSVR});
  benchmark::RunSpecifiedBenchmarks(&test_rep);

  for (auto& rep_test : TestCases) {
    std::string msg = std::string("\nTesting ") + rep_test.name + " Output\n";
    std::string banner(msg.size() - 1, '-');
    std::cout << banner << msg << banner << "\n";

    std::cerr << rep_test.err_stream.str();
    std::cout << rep_test.out_stream.str();

    internal::CheckCases(rep_test.error_cases, rep_test.err_stream);
    internal::CheckCases(rep_test.output_cases, rep_test.out_stream);

    std::cout << "\n";
  }
}
