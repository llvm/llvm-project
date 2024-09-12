// Test code starts here

//nftw test code:

namespace fs = std::filesystem;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    fs::path temp_path = fs::temp_directory_path();
    std::string temp_path_prefix = temp_path.string() + "/tmpdir.XXXXXX";
    // Use data() to get a writable string. mkdtemp doesn't write beyond the
    // allocated data.
    char* dir_name = mkdtemp(temp_path_prefix.data());
    _path = dir_name;
    fs::current_path(_path);
  }

  ~TemporaryDirectory() {
    fs::current_path(_path);
    fs::remove_all(_path);
  }
  const std::string GetDirectoryPath() { return _path.c_str(); }

 private:
  fs::path _path;
};

static void setupTestData() {
  fs::create_directories("sandbox");
  fs::create_directory("sandbox/owner_all_group_read_others_read_dir");
  fs::permissions("sandbox/owner_all_group_read_others_read_dir",
                  fs::perms::owner_all | fs::perms::group_read |
                      fs::perms::group_exec | fs::perms::others_read |
                      fs::perms::others_exec,
                  fs::perm_options::add);
  fs::create_directory(
      "sandbox/owner_all_group_read_others_read_dir/"
      "owner_read_group_read_others_read_dir");
  fs::permissions(
      "sandbox/owner_all_group_read_others_read_dir/"
      "owner_read_group_read_others_read_dir",
      fs::perms::owner_read | fs::perms::owner_exec | fs::perms::group_read |
          fs::perms::group_exec | fs::perms::others_read |
          fs::perms::others_exec,
      fs::perm_options::add);
  fs::create_directory("sandbox/no_perm_dir");
  fs::permissions("sandbox/no_perm_dir", fs::perms::none,
                  fs::perm_options::add);

  fs::create_symlink("invalid_target", "sandbox/sym1");
  fs::create_directory_symlink("owner_all_group_read_others_read_dir",
                               "sandbox/sym2");

  std::ofstream ofs("sandbox/file");  // create regular file
}

static bool isReadable(const fs::path& p) {
  std::error_code ec;  // For noexcept overload usage.
  auto perms = fs::status(p, ec).permissions();
  if ((perms & fs::perms::owner_read) != fs::perms::none &&
      (perms & fs::perms::group_read) != fs::perms::none &&
      (perms & fs::perms::others_read) != fs::perms::none) {
    return true;
  }
  return false;
}

static int checkNftw(const char* arg_fpath, const struct stat* statBuf,
                     int typeFlag, struct FTW* ftwbuf) {
  displayInfo(arg_fpath, statBuf, typeFlag, ftwbuf);
  if (arg_fpath == NULL) {
    std::cout << " fpath is null\n";
    return -1;
  }
  std::string fpath = arg_fpath;
  if (statBuf == NULL) {
    std::cout << " stat is null " << fpath << "\n";
    return -1;
  }

  const fs::path path = fpath;

  // status says we don't know the status of this file.
  if (typeFlag == FTW_NS || typeFlag == FTW_SLN) {
    struct stat sb;
    // Verify we can't stat the path.
    if (-1 != stat(arg_fpath, &sb)) {
      std::cout << "status doesn't match for " << arg_fpath << "\n";
      return -1;
    }
    return 0;
  }

  // If it is directory
  if (S_ISDIR(statBuf->st_mode)) {
    if (!fs::is_directory(fs::status(path))) {
      std::cout << "Is not directory> " << fpath << "\n";
      return -1;
    }
    if (isReadable(fpath)) {
      // It is readable, verify typeFlag is correct.
      if (typeFlag != FTW_D && typeFlag != FTW_DP) {
        std::cout << "typeFlag != FTW_D && typeFlag != FTW_DP " << fpath
                  << "\n";
        return -1;
      }
      return 0;
    }
    // It is not readable, and verify typeFlag is correct.
    if (typeFlag != FTW_DNR && typeFlag != FTW_D) {
      std::cout << "typeFlag != FTW_DNR && typeFlag != FTW_D " << fpath << "\n";
      return -1;
    }
    return 0;
  }

  // If it symlink, verify the filestatus and verify typeFlag is correct.
  if (S_ISLNK(statBuf->st_mode)) {
    if (!fs::is_symlink(fs::status(path))) {
      std::cout << "Is not symlink" << fpath << "\n";
      return -1;
    }
    if (FTW_SL != typeFlag) {
      std::cout << " FTW_SL != typeFlag " << fpath << "\n";
      return -1;
    }
    return 0;
  }

  if (!fs::is_regular_file(fs::status(path))) {
    std::cout << " is not a regular file " << fpath << "\n";
    return -1;
  }
  if (FTW_F != typeFlag) {
    std::cout << " FTW_SL != typeFlag " << fpath << "\n";
    return -1;
  }
  return 0; /* To tell llvm_libc_nftw() to continue */
}

static void testNftw() {
  std::cout << std::endl << "Calling testNftw: " << std::endl;
  TemporaryDirectory tmpDir = TemporaryDirectory();
  setupTestData();
  int flags = 0;
  llvm_libc_nftw(tmpDir.GetDirectoryPath(), checkNftw, 128, flags);
  std::cout << "All testNftw tests have passed: " << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "ftw called with args: " << argv[1] << std::endl;

  int flags = 0;

  if (argc > 2) {
    if (strchr(argv[2], 'p') != NULL) flags |= FTW_PHYS;
    if (strchr(argv[2], 'd') != NULL) flags |= FTW_DEPTH;
  } else {
    flags |= FTW_DEPTH;
  }

  std::cout << "Calling nftw: " << std::endl;
  if (nftw((argc < 2) ? "." : argv[1], displayInfo, 20, flags) == -1) {
    perror("nftw");
    exit(EXIT_FAILURE);
  }

  std::cout << "Calling llvm_libc_nftw: " << std::endl;
  std::string_view dirPath(argv[1]);
  if (llvm_libc_nftw((argc < 2) ? "." : argv[1], displayInfo, 20, flags) ==
      -1) {
    perror("llvm_libc_nftw");
    exit(EXIT_FAILURE);
  }

  // Unit tests for FTW.
  testNftw();

  exit(EXIT_SUCCESS);
}

// ftw test code:


static int display_info(const char *filePath, const struct stat *sb, int tflag,
                        struct FTW *ftwbuf) {
  printf("%-3s %2d ",
         (tflag == FTW_D)     ? "d"
         : (tflag == FTW_DNR) ? "dnr"
         : (tflag == FTW_DP)  ? "dp"
         : (tflag == FTW_F)   ? "f"
         : (tflag == FTW_NS)  ? "ns"
         : (tflag == FTW_SL)  ? "sl"
         : (tflag == FTW_SLN) ? "sln"
                              : "???",
         ftwbuf->level);

  if (tflag == FTW_NS)
    printf("-------");
  else
    printf("%7jd", (intmax_t)sb->st_size);

  printf("   %-40s %d %s\n", filePath, ftwbuf->base, filePath + ftwbuf->base);

  return 0; /* To tell llvm_libc_nftw() to continue */
}

int main(int argc, char *argv[]) {
  std::cout << "ftw called with args: " << argv[1] << std::endl;

  int flags = 0;

  if (argc > 2) {
    if (strchr(argv[2], 'p') != NULL) flags |= FTW_PHYS;
    if (strchr(argv[2], 'd') != NULL) flags |= FTW_DEPTH;
  } else {
    flags |= FTW_DEPTH;
  }

  std::cout << "Calling nftw: " << std::endl;
  if (ftw((argc < 2) ? "." : argv[1], display_info, 20) == -1) {
    perror("nftw");
    exit(EXIT_FAILURE);
  }

  std::cout << "Calling llvm_libc_nftw: " << std::endl;
  std::string_view dirPath(argv[1]);
  if (llvm_libc_ftw((argc < 2) ? "." : argv[1], display_info, 20) == -1) {
    perror("llvm_libc_nftw");
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}

