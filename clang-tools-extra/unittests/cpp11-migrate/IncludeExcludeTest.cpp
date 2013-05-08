#include "Core/IncludeExcludeInfo.h"
#include "gtest/gtest.h"
#include "llvm/Support/Path.h"
#include <fstream>

TEST(IncludeExcludeTest, ParseString) {
  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromString(
      /*include=*/ "a,b/b2,c/c2",
      /*exclude=*/ "a/af.cpp,a/a2,b/b2/b2f.cpp,c/c2");

  ASSERT_EQ(Err, llvm::error_code::success());

  // If the file does not appear on the include list then it is not safe to
  // transform. Files are not safe to transform by default.
  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/dir/f.cpp"));

  // If the file appears on only the include list then it is safe to transform.
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/dir/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("b/b2/f.cpp"));

  // If the file appears on both the include or exclude list then it is not
  // safe to transform.
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/dir/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/b2/b2f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("c/c2/c3/f.cpp"));
}

// Utility for creating and filling files with data for IncludeExcludeFileTest
// tests.
struct InputFiles {

  // This function uses fatal assertions. The caller is responsible for making
  // sure fatal assertions propagate.
  void CreateFiles(bool UnixMode) {
    IncludeDataPath = llvm::sys::Path::GetTemporaryDirectory();
    ExcludeDataPath = IncludeDataPath;

    ASSERT_FALSE(IncludeDataPath.createTemporaryFileOnDisk());
    std::ofstream IncludeDataFile(IncludeDataPath.c_str());
    ASSERT_TRUE(IncludeDataFile.good());
    for (unsigned i = 0; i < sizeof(IncludeData)/sizeof(char*); ++i) {
      IncludeDataFile << IncludeData[i] << (UnixMode ? "\n" : "\r\n");
    }

    ASSERT_FALSE(ExcludeDataPath.createTemporaryFileOnDisk());
    std::ofstream ExcludeDataFile(ExcludeDataPath.c_str());
    ASSERT_TRUE(ExcludeDataFile.good());
    for (unsigned i = 0; i < sizeof(ExcludeData)/sizeof(char*); ++i) {
      ExcludeDataFile << ExcludeData[i] << (UnixMode ? "\n" : "\r\n");;
    }
  }

  static const char *IncludeData[3];
  static const char *ExcludeData[4];

  llvm::sys::Path IncludeDataPath;
  llvm::sys::Path ExcludeDataPath;
};

const char *InputFiles::IncludeData[3] = { "a", "b/b2", "c/c2" };
const char *InputFiles::ExcludeData[4] = { "a/af.cpp", "a/a2", "b/b2/b2f.cpp",
                                           "c/c2" };

TEST(IncludeExcludeFileTest, UNIXFile) {
  InputFiles UnixFiles;
  ASSERT_NO_FATAL_FAILURE(UnixFiles.CreateFiles(/* UnixMode= */true));

  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromFile(
      UnixFiles.IncludeDataPath.c_str(), UnixFiles.ExcludeDataPath.c_str());

  ASSERT_EQ(Err, llvm::error_code::success());

  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
}

TEST(IncludeExcludeFileTest, DOSFile) {
  InputFiles DOSFiles;
  ASSERT_NO_FATAL_FAILURE(DOSFiles.CreateFiles(/* UnixMode= */false));

  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromFile(
      DOSFiles.IncludeDataPath.c_str(), DOSFiles.ExcludeDataPath.c_str());

  ASSERT_EQ(Err, llvm::error_code::success());

  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
}
