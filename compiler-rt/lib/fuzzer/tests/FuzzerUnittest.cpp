// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Avoid ODR violations (LibFuzzer is built without ASan and this test is built
// with ASan) involving C++ standard library types when using libcxx.
#define _LIBCPP_HAS_NO_ASAN

// Do not attempt to use LLVM ostream etc from gtest.
#define GTEST_NO_LLVM_SUPPORT 1

#include "FuzzerCorpus.h"
#include "FuzzerInternal.h"
#include "FuzzerMerge.h"
#include "FuzzerMutate.h"
#include "FuzzerRandom.h"
#include "FuzzerTracePC.h"
#include "gtest/gtest.h"
#include <memory>
#include <set>
#include <sstream>

using namespace fuzzer;

// For now, have LLVMFuzzerTestOneInput just to make it link.
// Later we may want to make unittests that actually call LLVMFuzzerTestOneInput.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  abort();
}

TEST(Fuzzer, Basename) {
  EXPECT_EQ(Basename("foo/bar"), "bar");
  EXPECT_EQ(Basename("bar"), "bar");
  EXPECT_EQ(Basename("/bar"), "bar");
  EXPECT_EQ(Basename("foo/x"), "x");
  EXPECT_EQ(Basename("foo/"), "");
#if LIBFUZZER_WINDOWS
  EXPECT_EQ(Basename("foo\\bar"), "bar");
  EXPECT_EQ(Basename("foo\\bar/baz"), "baz");
  EXPECT_EQ(Basename("\\bar"), "bar");
  EXPECT_EQ(Basename("foo\\x"), "x");
  EXPECT_EQ(Basename("foo\\"), "");
#endif
}

TEST(Fuzzer, Hash) {
  uint8_t A[] = {'a', 'b', 'c'};
  fuzzer::Unit U(A, A + sizeof(A));
  EXPECT_EQ("a9993e364706816aba3e25717850c26c9cd0d89d", fuzzer::Hash(U));
  U.push_back('d');
  EXPECT_EQ("81fe8bfe87576c3ecb22426f8e57847382917acf", fuzzer::Hash(U));
}

TEST(FuzzerDictionary, ParseOneDictionaryEntry) {
  Unit U;
  EXPECT_FALSE(ParseOneDictionaryEntry("", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry(" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("\t  ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  zz\" ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \"zz ", &U));
  EXPECT_FALSE(ParseOneDictionaryEntry("  \"\" ", &U));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"a\"", &U));
  EXPECT_EQ(U, Unit({'a'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"abc\"", &U));
  EXPECT_EQ(U, Unit({'a', 'b', 'c'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("abc=\"abc\"", &U));
  EXPECT_EQ(U, Unit({'a', 'b', 'c'}));
  EXPECT_FALSE(ParseOneDictionaryEntry("\"\\\"", &U));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\\\\"", &U));
  EXPECT_EQ(U, Unit({'\\'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\xAB\"", &U));
  EXPECT_EQ(U, Unit({0xAB}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\xABz\\xDE\"", &U));
  EXPECT_EQ(U, Unit({0xAB, 'z', 0xDE}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"#\"", &U));
  EXPECT_EQ(U, Unit({'#'}));
  EXPECT_TRUE(ParseOneDictionaryEntry("\"\\\"\"", &U));
  EXPECT_EQ(U, Unit({'"'}));
}

TEST(FuzzerDictionary, ParseDictionaryFile) {
  Vector<Unit> Units;
  EXPECT_FALSE(ParseDictionaryFile("zzz\n", &Units));
  EXPECT_FALSE(ParseDictionaryFile("", &Units));
  EXPECT_TRUE(ParseDictionaryFile("\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("#zzzz a b c d\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile(" #zzzz\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("  #zzzz\n", &Units));
  EXPECT_EQ(Units.size(), 0U);
  EXPECT_TRUE(ParseDictionaryFile("  #zzzz\naaa=\"aa\"", &Units));
  EXPECT_EQ(Units, Vector<Unit>({Unit({'a', 'a'})}));
  EXPECT_TRUE(
      ParseDictionaryFile("  #zzzz\naaa=\"aa\"\n\nabc=\"abc\"", &Units));
  EXPECT_EQ(Units,
            Vector<Unit>({Unit({'a', 'a'}), Unit({'a', 'b', 'c'})}));
}

TEST(FuzzerUtil, Base64) {
  EXPECT_EQ("", Base64({}));
  EXPECT_EQ("YQ==", Base64({'a'}));
  EXPECT_EQ("eA==", Base64({'x'}));
  EXPECT_EQ("YWI=", Base64({'a', 'b'}));
  EXPECT_EQ("eHk=", Base64({'x', 'y'}));
  EXPECT_EQ("YWJj", Base64({'a', 'b', 'c'}));
  EXPECT_EQ("eHl6", Base64({'x', 'y', 'z'}));
  EXPECT_EQ("YWJjeA==", Base64({'a', 'b', 'c', 'x'}));
  EXPECT_EQ("YWJjeHk=", Base64({'a', 'b', 'c', 'x', 'y'}));
  EXPECT_EQ("YWJjeHl6", Base64({'a', 'b', 'c', 'x', 'y', 'z'}));
}

TEST(Corpus, Distribution) {
  DataFlowTrace DFT;
  Random Rand(0);
  struct EntropicOptions Entropic = {false, 0xFF, 100, false};
  std::unique_ptr<InputCorpus> C(new InputCorpus("", Entropic));
  size_t N = 10;
  size_t TriesPerUnit = 1<<16;
  for (size_t i = 0; i < N; i++)
    C->AddToCorpus(Unit{static_cast<uint8_t>(i)}, /*NumFeatures*/ 1,
                   /*MayDeleteFile*/ false, /*HasFocusFunction*/ false,
                   /*ForceAddToCorpus*/ false,
                   /*TimeOfUnit*/ std::chrono::microseconds(0),
                   /*FeatureSet*/ {}, DFT,
                   /*BaseII*/ nullptr);

  Vector<size_t> Hist(N);
  for (size_t i = 0; i < N * TriesPerUnit; i++) {
    Hist[C->ChooseUnitIdxToMutate(Rand)]++;
  }
  for (size_t i = 0; i < N; i++) {
    // A weak sanity check that every unit gets invoked.
    EXPECT_GT(Hist[i], TriesPerUnit / N / 3);
  }
}

template <typename T> void EQ(const Vector<T> &A, const Vector<T> &B) {
  EXPECT_EQ(A, B);
}

template <typename T> void EQ(const Set<T> &A, const Vector<T> &B) {
  EXPECT_EQ(A, Set<T>(B.begin(), B.end()));
}

void EQ(const Vector<MergeFileInfo> &A, const Vector<std::string> &B) {
  Set<std::string> a;
  for (const auto &File : A)
    a.insert(File.Name);
  Set<std::string> b(B.begin(), B.end());
  EXPECT_EQ(a, b);
}

#define TRACED_EQ(A, ...)                                                      \
  {                                                                            \
    SCOPED_TRACE(#A);                                                          \
    EQ(A, __VA_ARGS__);                                                        \
  }

TEST(Merger, Parse) {
  Merger M;

  const char *kInvalidInputs[] = {
      // Bad file numbers
      "",
      "x",
      "0\n0",
      "3\nx",
      "2\n3",
      "2\n2",
      // Bad file names
      "2\n2\nA\n",
      "2\n2\nA\nB\nC\n",
      // Unknown markers
      "2\n1\nA\nSTARTED 0\nBAD 0 0x0",
      // Bad file IDs
      "1\n1\nA\nSTARTED 1",
      "2\n1\nA\nSTARTED 0\nFT 1 0x0",
  };
  for (auto S : kInvalidInputs) {
    SCOPED_TRACE(S);
    EXPECT_FALSE(M.Parse(S, false));
  }

  // Parse initial control file
  EXPECT_TRUE(M.Parse("1\n0\nAA\n", false));
  ASSERT_EQ(M.Files.size(), 1U);
  EXPECT_EQ(M.NumFilesInFirstCorpus, 0U);
  EXPECT_EQ(M.Files[0].Name, "AA");
  EXPECT_TRUE(M.LastFailure.empty());
  EXPECT_EQ(M.FirstNotProcessedFile, 0U);

  // Parse control file that failed on first attempt
  EXPECT_TRUE(M.Parse("2\n1\nAA\nBB\nSTARTED 0 42\n", false));
  ASSERT_EQ(M.Files.size(), 2U);
  EXPECT_EQ(M.NumFilesInFirstCorpus, 1U);
  EXPECT_EQ(M.Files[0].Name, "AA");
  EXPECT_EQ(M.Files[1].Name, "BB");
  EXPECT_EQ(M.LastFailure, "AA");
  EXPECT_EQ(M.FirstNotProcessedFile, 1U);

  // Parse control file that failed on later attempt
  EXPECT_TRUE(M.Parse("3\n1\nAA\nBB\nC\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1002\n"
                      "",
                      true));
  ASSERT_EQ(M.Files.size(), 3U);
  EXPECT_EQ(M.NumFilesInFirstCorpus, 1U);
  EXPECT_EQ(M.Files[0].Name, "AA");
  EXPECT_EQ(M.Files[0].Size, 1000U);
  EXPECT_EQ(M.Files[1].Name, "BB");
  EXPECT_EQ(M.Files[1].Size, 1001U);
  EXPECT_EQ(M.Files[2].Name, "C");
  EXPECT_EQ(M.Files[2].Size, 1002U);
  EXPECT_EQ(M.LastFailure, "C");
  EXPECT_EQ(M.FirstNotProcessedFile, 3U);
  TRACED_EQ(M.Files[0].Features, {1, 2, 3});
  TRACED_EQ(M.Files[1].Features, {4, 5, 6});

  // Parse control file without features or PCs
  EXPECT_TRUE(M.Parse("2\n0\nAA\nBB\n"
                      "STARTED 0 1000\n"
                      "FT 0\n"
                      "COV 0\n"
                      "STARTED 1 1001\n"
                      "FT 1\n"
                      "COV 1\n"
                      "",
                      true));
  ASSERT_EQ(M.Files.size(), 2U);
  EXPECT_EQ(M.NumFilesInFirstCorpus, 0U);
  EXPECT_TRUE(M.LastFailure.empty());
  EXPECT_EQ(M.FirstNotProcessedFile, 2U);
  EXPECT_TRUE(M.Files[0].Features.empty());
  EXPECT_TRUE(M.Files[0].Cov.empty());
  EXPECT_TRUE(M.Files[1].Features.empty());
  EXPECT_TRUE(M.Files[1].Cov.empty());

  // Parse features and PCs
  EXPECT_TRUE(M.Parse("3\n2\nAA\nBB\nC\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "COV 0 11 12 13\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6\n"
                      "COV 1 7 8 9\n"
                      "STARTED 2 1002\n"
                      "FT 2 6 1 3\n"
                      "COV 2 16 11 13\n"
                      "",
                      true));
  ASSERT_EQ(M.Files.size(), 3U);
  EXPECT_EQ(M.NumFilesInFirstCorpus, 2U);
  EXPECT_TRUE(M.LastFailure.empty());
  EXPECT_EQ(M.FirstNotProcessedFile, 3U);
  TRACED_EQ(M.Files[0].Features, {1, 2, 3});
  TRACED_EQ(M.Files[0].Cov, {11, 12, 13});
  TRACED_EQ(M.Files[1].Features, {4, 5, 6});
  TRACED_EQ(M.Files[1].Cov, {7, 8, 9});
  TRACED_EQ(M.Files[2].Features, {1, 3, 6});
  TRACED_EQ(M.Files[2].Cov, {16});
}

TEST(Merger, Merge) {
  Merger M;
  Set<uint32_t> Features, NewFeatures;
  Set<uint32_t> Cov, NewCov;
  Vector<std::string> NewFiles;

  // Adds new files and features
  EXPECT_TRUE(M.Parse("3\n0\nA\nB\nC\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1002\n"
                      "FT 2 6 1 3\n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 6U);
  TRACED_EQ(M.Files, {"A", "B", "C"});
  TRACED_EQ(NewFiles, {"A", "B"});
  TRACED_EQ(NewFeatures, {1, 2, 3, 4, 5, 6});

  // Doesn't return features or files in the initial corpus.
  EXPECT_TRUE(M.Parse("3\n1\nA\nB\nC\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1002\n"
                      "FT 2 6 1 3\n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 3U);
  TRACED_EQ(M.Files, {"A", "B", "C"});
  TRACED_EQ(NewFiles, {"B"});
  TRACED_EQ(NewFeatures, {4, 5, 6});

  // No new features, so no new files
  EXPECT_TRUE(M.Parse("3\n2\nA\nB\nC\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1002\n"
                      "FT 2 6 1 3\n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 0U);
  TRACED_EQ(M.Files, {"A", "B", "C"});
  TRACED_EQ(NewFiles, {});
  TRACED_EQ(NewFeatures, {});

  // Can pass initial features and coverage.
  Features = {1, 2, 3};
  Cov = {};
  EXPECT_TRUE(M.Parse("2\n0\nA\nB\n"
                      "STARTED 0 1000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6\n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 3U);
  TRACED_EQ(M.Files, {"A", "B"});
  TRACED_EQ(NewFiles, {"B"});
  TRACED_EQ(NewFeatures, {4, 5, 6});
  Features.clear();
  Cov.clear();

  // Parse smaller files first
  EXPECT_TRUE(M.Parse("3\n0\nA\nB\nC\n"
                      "STARTED 0 2000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1001\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1002\n"
                      "FT 2 6 1 3 \n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 6U);
  TRACED_EQ(M.Files, {"B", "C", "A"});
  TRACED_EQ(NewFiles, {"B", "C", "A"});
  TRACED_EQ(NewFeatures, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE(M.Parse("4\n0\nA\nB\nC\nD\n"
                      "STARTED 0 2000\n"
                      "FT 0 1 2 3\n"
                      "STARTED 1 1101\n"
                      "FT 1 4 5 6 \n"
                      "STARTED 2 1102\n"
                      "FT 2 6 1 3 100 \n"
                      "STARTED 3 1000\n"
                      "FT 3 1  \n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 7U);
  TRACED_EQ(M.Files, {"A", "B", "C", "D"});
  TRACED_EQ(NewFiles, {"D", "B", "C", "A"});
  TRACED_EQ(NewFeatures, {1, 2, 3, 4, 5, 6, 100});

  // For same sized file, parse more features first
  EXPECT_TRUE(M.Parse("4\n1\nA\nB\nC\nD\n"
                      "STARTED 0 2000\n"
                      "FT 0 4 5 6 7 8\n"
                      "STARTED 1 1100\n"
                      "FT 1 1 2 3 \n"
                      "STARTED 2 1100\n"
                      "FT 2 2 3 \n"
                      "STARTED 3 1000\n"
                      "FT 3 1  \n"
                      "",
                      true));
  EXPECT_EQ(M.Merge(Features, &NewFeatures, Cov, &NewCov, &NewFiles), 3U);
  TRACED_EQ(M.Files, {"A", "B", "C", "D"});
  TRACED_EQ(NewFiles, {"D", "B"});
  TRACED_EQ(NewFeatures, {1, 2, 3});
}

#undef TRACED_EQ

TEST(DFT, BlockCoverage) {
  BlockCoverage Cov;
  // Assuming C0 has 5 instrumented blocks,
  // C1: 7 blocks, C2: 4, C3: 9, C4 never covered, C5: 15,

  // Add C0
  EXPECT_TRUE(Cov.AppendCoverage("C0 5\n"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(0, 1), 0U);  // not seen this BB yet.
  EXPECT_EQ(Cov.GetCounter(0, 5), 0U);  // BB ID out of bounds.
  EXPECT_EQ(Cov.GetCounter(1, 0), 0U);  // not seen this function yet.

  EXPECT_EQ(Cov.GetNumberOfBlocks(0), 5U);
  EXPECT_EQ(Cov.GetNumberOfCoveredBlocks(0), 1U);
  EXPECT_EQ(Cov.GetNumberOfBlocks(1), 0U);

  // Various errors.
  EXPECT_FALSE(Cov.AppendCoverage("C0\n"));  // No total number.
  EXPECT_FALSE(Cov.AppendCoverage("C0 7\n"));  // No total number.
  EXPECT_FALSE(Cov.AppendCoverage("CZ\n"));  // Wrong function number.
  EXPECT_FALSE(Cov.AppendCoverage("C1 7 7"));  // BB ID is too big.
  EXPECT_FALSE(Cov.AppendCoverage("C1 100 7")); // BB ID is too big.

  // Add C0 more times.
  EXPECT_TRUE(Cov.AppendCoverage("C0 5\n"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 2U);
  EXPECT_TRUE(Cov.AppendCoverage("C0 1 2 5\n"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 3U);
  EXPECT_EQ(Cov.GetCounter(0, 1), 1U);
  EXPECT_EQ(Cov.GetCounter(0, 2), 1U);
  EXPECT_EQ(Cov.GetCounter(0, 3), 0U);
  EXPECT_EQ(Cov.GetCounter(0, 4), 0U);
  EXPECT_EQ(Cov.GetNumberOfCoveredBlocks(0), 3U);
  EXPECT_TRUE(Cov.AppendCoverage("C0 1 3 4 5\n"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 4U);
  EXPECT_EQ(Cov.GetCounter(0, 1), 2U);
  EXPECT_EQ(Cov.GetCounter(0, 2), 1U);
  EXPECT_EQ(Cov.GetCounter(0, 3), 1U);
  EXPECT_EQ(Cov.GetCounter(0, 4), 1U);
  EXPECT_EQ(Cov.GetNumberOfCoveredBlocks(0), 5U);

  EXPECT_TRUE(Cov.AppendCoverage("C1 7\nC2 4\nC3 9\nC5 15\nC0 5\n"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 5U);
  EXPECT_EQ(Cov.GetCounter(1, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(2, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(3, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(4, 0), 0U);
  EXPECT_EQ(Cov.GetCounter(5, 0), 1U);

  EXPECT_TRUE(Cov.AppendCoverage("C3 4 5 9\nC5 11 12 15"));
  EXPECT_EQ(Cov.GetCounter(0, 0), 5U);
  EXPECT_EQ(Cov.GetCounter(1, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(2, 0), 1U);
  EXPECT_EQ(Cov.GetCounter(3, 0), 2U);
  EXPECT_EQ(Cov.GetCounter(3, 4), 1U);
  EXPECT_EQ(Cov.GetCounter(3, 5), 1U);
  EXPECT_EQ(Cov.GetCounter(3, 6), 0U);
  EXPECT_EQ(Cov.GetCounter(4, 0), 0U);
  EXPECT_EQ(Cov.GetCounter(5, 0), 2U);
  EXPECT_EQ(Cov.GetCounter(5, 10), 0U);
  EXPECT_EQ(Cov.GetCounter(5, 11), 1U);
  EXPECT_EQ(Cov.GetCounter(5, 12), 1U);
}

TEST(DFT, FunctionWeights) {
  BlockCoverage Cov;
  // unused function gets zero weight.
  EXPECT_TRUE(Cov.AppendCoverage("C0 5\n"));
  auto Weights = Cov.FunctionWeights(2);
  EXPECT_GT(Weights[0], 0.);
  EXPECT_EQ(Weights[1], 0.);

  // Less frequently used function gets less weight.
  Cov.clear();
  EXPECT_TRUE(Cov.AppendCoverage("C0 5\nC1 5\nC1 5\n"));
  Weights = Cov.FunctionWeights(2);
  EXPECT_GT(Weights[0], Weights[1]);

  // A function with more uncovered blocks gets more weight.
  Cov.clear();
  EXPECT_TRUE(Cov.AppendCoverage("C0 1 2 3 5\nC1 2 4\n"));
  Weights = Cov.FunctionWeights(2);
  EXPECT_GT(Weights[1], Weights[0]);

  // A function with DFT gets more weight than the function w/o DFT.
  Cov.clear();
  EXPECT_TRUE(Cov.AppendCoverage("F1 111\nC0 3\nC1 1 2 3\n"));
  Weights = Cov.FunctionWeights(2);
  EXPECT_GT(Weights[1], Weights[0]);
}


TEST(Fuzzer, ForEachNonZeroByte) {
  const size_t N = 64;
  alignas(64) uint8_t Ar[N + 8] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 3, 0, 4, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 5, 0, 6, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
  };
  typedef Vector<std::pair<size_t, uint8_t> > Vec;
  Vec Res, Expected;
  auto CB = [&](size_t FirstFeature, size_t Idx, uint8_t V) {
    Res.push_back({FirstFeature + Idx, V});
  };
  ForEachNonZeroByte(Ar, Ar + N, 100, CB);
  Expected = {{108, 1}, {109, 2}, {118, 3}, {120, 4},
              {135, 5}, {137, 6}, {146, 7}, {163, 8}};
  EXPECT_EQ(Res, Expected);

  Res.clear();
  ForEachNonZeroByte(Ar + 9, Ar + N, 109, CB);
  Expected = {          {109, 2}, {118, 3}, {120, 4},
              {135, 5}, {137, 6}, {146, 7}, {163, 8}};
  EXPECT_EQ(Res, Expected);

  Res.clear();
  ForEachNonZeroByte(Ar + 9, Ar + N - 9, 109, CB);
  Expected = {          {109, 2}, {118, 3}, {120, 4},
              {135, 5}, {137, 6}, {146, 7}};
  EXPECT_EQ(Res, Expected);
}

// FuzzerCommand unit tests. The arguments in the two helper methods below must
// match.
static void makeCommandArgs(Vector<std::string> *ArgsToAdd) {
  assert(ArgsToAdd);
  ArgsToAdd->clear();
  ArgsToAdd->push_back("foo");
  ArgsToAdd->push_back("-bar=baz");
  ArgsToAdd->push_back("qux");
  ArgsToAdd->push_back(Command::ignoreRemainingArgs());
  ArgsToAdd->push_back("quux");
  ArgsToAdd->push_back("-grault=garply");
}

static std::string makeCmdLine(const char *separator, const char *suffix) {
  std::string CmdLine("foo -bar=baz qux ");
  if (strlen(separator) != 0) {
    CmdLine += separator;
    CmdLine += " ";
  }
  CmdLine += Command::ignoreRemainingArgs();
  CmdLine += " quux -grault=garply";
  if (strlen(suffix) != 0) {
    CmdLine += " ";
    CmdLine += suffix;
  }
  return CmdLine;
}

TEST(FuzzerCommand, Create) {
  std::string CmdLine;

  // Default constructor
  Command DefaultCmd;

  CmdLine = DefaultCmd.toString();
  EXPECT_EQ(CmdLine, "");

  // Explicit constructor
  Vector<std::string> ArgsToAdd;
  makeCommandArgs(&ArgsToAdd);
  Command InitializedCmd(ArgsToAdd);

  CmdLine = InitializedCmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));

  // Compare each argument
  auto InitializedArgs = InitializedCmd.getArguments();
  auto i = ArgsToAdd.begin();
  auto j = InitializedArgs.begin();
  while (i != ArgsToAdd.end() && j != InitializedArgs.end()) {
    EXPECT_EQ(*i++, *j++);
  }
  EXPECT_EQ(i, ArgsToAdd.end());
  EXPECT_EQ(j, InitializedArgs.end());

  // Copy constructor
  Command CopiedCmd(InitializedCmd);

  CmdLine = CopiedCmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));

  // Assignment operator
  Command AssignedCmd;
  AssignedCmd = CopiedCmd;

  CmdLine = AssignedCmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));
}

TEST(FuzzerCommand, ModifyArguments) {
  Vector<std::string> ArgsToAdd;
  makeCommandArgs(&ArgsToAdd);
  Command Cmd;
  std::string CmdLine;

  Cmd.addArguments(ArgsToAdd);
  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));

  Cmd.addArgument("waldo");
  EXPECT_TRUE(Cmd.hasArgument("waldo"));

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("waldo", ""));

  Cmd.removeArgument("waldo");
  EXPECT_FALSE(Cmd.hasArgument("waldo"));

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));
}

TEST(FuzzerCommand, ModifyFlags) {
  Vector<std::string> ArgsToAdd;
  makeCommandArgs(&ArgsToAdd);
  Command Cmd(ArgsToAdd);
  std::string Value, CmdLine;
  ASSERT_FALSE(Cmd.hasFlag("fred"));

  Value = Cmd.getFlagValue("fred");
  EXPECT_EQ(Value, "");

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));

  Cmd.addFlag("fred", "plugh");
  EXPECT_TRUE(Cmd.hasFlag("fred"));

  Value = Cmd.getFlagValue("fred");
  EXPECT_EQ(Value, "plugh");

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("-fred=plugh", ""));

  Cmd.removeFlag("fred");
  EXPECT_FALSE(Cmd.hasFlag("fred"));

  Value = Cmd.getFlagValue("fred");
  EXPECT_EQ(Value, "");

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ""));
}

TEST(FuzzerCommand, SetOutput) {
  Vector<std::string> ArgsToAdd;
  makeCommandArgs(&ArgsToAdd);
  Command Cmd(ArgsToAdd);
  std::string CmdLine;
  ASSERT_FALSE(Cmd.hasOutputFile());
  ASSERT_FALSE(Cmd.isOutAndErrCombined());

  Cmd.combineOutAndErr(true);
  EXPECT_TRUE(Cmd.isOutAndErrCombined());

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", "2>&1"));

  Cmd.combineOutAndErr(false);
  EXPECT_FALSE(Cmd.isOutAndErrCombined());

  Cmd.setOutputFile("xyzzy");
  EXPECT_TRUE(Cmd.hasOutputFile());

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ">xyzzy"));

  Cmd.setOutputFile("thud");
  EXPECT_TRUE(Cmd.hasOutputFile());

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ">thud"));

  Cmd.combineOutAndErr();
  EXPECT_TRUE(Cmd.isOutAndErrCombined());

  CmdLine = Cmd.toString();
  EXPECT_EQ(CmdLine, makeCmdLine("", ">thud 2>&1"));
}

TEST(Entropic, UpdateFrequency) {
  const size_t One = 1, Two = 2;
  const size_t FeatIdx1 = 0, FeatIdx2 = 42, FeatIdx3 = 12, FeatIdx4 = 26;
  size_t Index;
  // Create input corpus with default entropic configuration
  struct EntropicOptions Entropic = {true, 0xFF, 100, false};
  std::unique_ptr<InputCorpus> C(new InputCorpus("", Entropic));
  std::unique_ptr<InputInfo> II(new InputInfo());

  C->AddRareFeature(FeatIdx1);
  C->UpdateFeatureFrequency(II.get(), FeatIdx1);
  EXPECT_EQ(II->FeatureFreqs.size(), One);
  C->AddRareFeature(FeatIdx2);
  C->UpdateFeatureFrequency(II.get(), FeatIdx1);
  C->UpdateFeatureFrequency(II.get(), FeatIdx2);
  EXPECT_EQ(II->FeatureFreqs.size(), Two);
  EXPECT_EQ(II->FeatureFreqs[0].second, 2);
  EXPECT_EQ(II->FeatureFreqs[1].second, 1);

  C->AddRareFeature(FeatIdx3);
  C->AddRareFeature(FeatIdx4);
  C->UpdateFeatureFrequency(II.get(), FeatIdx3);
  C->UpdateFeatureFrequency(II.get(), FeatIdx3);
  C->UpdateFeatureFrequency(II.get(), FeatIdx3);
  C->UpdateFeatureFrequency(II.get(), FeatIdx4);

  for (Index = 1; Index < II->FeatureFreqs.size(); Index++)
    EXPECT_LT(II->FeatureFreqs[Index - 1].first, II->FeatureFreqs[Index].first);

  II->DeleteFeatureFreq(FeatIdx3);
  for (Index = 1; Index < II->FeatureFreqs.size(); Index++)
    EXPECT_LT(II->FeatureFreqs[Index - 1].first, II->FeatureFreqs[Index].first);
}

double SubAndSquare(double X, double Y) {
  double R = X - Y;
  R = R * R;
  return R;
}

TEST(Entropic, ComputeEnergy) {
  const double Precision = 0.01;
  struct EntropicOptions Entropic = {true, 0xFF, 100, false};
  std::unique_ptr<InputCorpus> C(new InputCorpus("", Entropic));
  std::unique_ptr<InputInfo> II(new InputInfo());
  Vector<std::pair<uint32_t, uint16_t>> FeatureFreqs = {{1, 3}, {2, 3}, {3, 3}};
  II->FeatureFreqs = FeatureFreqs;
  II->NumExecutedMutations = 0;
  II->UpdateEnergy(4, false, std::chrono::microseconds(0));
  EXPECT_LT(SubAndSquare(II->Energy, 1.450805), Precision);

  II->NumExecutedMutations = 9;
  II->UpdateEnergy(5, false, std::chrono::microseconds(0));
  EXPECT_LT(SubAndSquare(II->Energy, 1.525496), Precision);

  II->FeatureFreqs[0].second++;
  II->FeatureFreqs.push_back(std::pair<uint32_t, uint16_t>(42, 6));
  II->NumExecutedMutations = 20;
  II->UpdateEnergy(10, false, std::chrono::microseconds(0));
  EXPECT_LT(SubAndSquare(II->Energy, 1.792831), Precision);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
