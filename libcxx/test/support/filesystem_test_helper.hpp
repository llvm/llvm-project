#ifndef FILESYSTEM_TEST_HELPER_HPP
#define FILESYSTEM_TEST_HELPER_HPP

#include <experimental/filesystem>
#include <cassert>
#include <cstdio> // for printf
#include <string>
#include <fstream>
#include <random>
#include <chrono>

namespace fs = std::experimental::filesystem;

// static test helpers

#ifndef LIBCXX_FILESYSTEM_STATIC_TEST_ROOT
#warning "STATIC TESTS DISABLED"
#else // LIBCXX_FILESYSTEM_STATIC_TEST_ROOT

namespace StaticEnv {

inline fs::path makePath(fs::path const& p) {
    // env_path is expected not to contain symlinks.
    static const fs::path env_path = LIBCXX_FILESYSTEM_STATIC_TEST_ROOT;
    return env_path / p;
}

static const fs::path Root = LIBCXX_FILESYSTEM_STATIC_TEST_ROOT;

static const fs::path TestFileList[] = {
        makePath("empty_file"),
        makePath("non_empty_file"),
        makePath("dir1/file1"),
        makePath("dir1/file2")
};
const std::size_t TestFileListSize = sizeof(TestFileList) / sizeof(fs::path);

static const fs::path TestDirList[] = {
        makePath("dir1"),
        makePath("dir1/dir2"),
        makePath("dir1/dir2/dir3")
};
const std::size_t TestDirListSize = sizeof(TestDirList) / sizeof(fs::path);

static const fs::path File          = TestFileList[0];
static const fs::path Dir           = TestDirList[0];
static const fs::path Dir2          = TestDirList[1];
static const fs::path Dir3          = TestDirList[2];
static const fs::path SymlinkToFile = makePath("symlink_to_empty_file");
static const fs::path SymlinkToDir  = makePath("symlink_to_dir");
static const fs::path BadSymlink    = makePath("bad_symlink");
static const fs::path DNE           = makePath("DNE");
static const fs::path EmptyFile     = TestFileList[0];
static const fs::path NonEmptyFile  = TestFileList[1];
static const fs::path CharFile      = "/dev/null"; // Hopefully this exists

static const fs::path DirIterationList[] = {
    makePath("dir1/dir2"),
    makePath("dir1/file1"),
    makePath("dir1/file2")
};
const std::size_t DirIterationListSize = sizeof(DirIterationList)
                                        / sizeof(fs::path);

static const fs::path DirIterationListDepth1[] = {
    makePath("dir1/dir2/afile3"),
    makePath("dir1/dir2/dir3"),
    makePath("dir1/dir2/symlink_to_dir3"),
    makePath("dir1/dir2/file4"),
};

static const fs::path RecDirIterationList[] = {
    makePath("dir1/dir2"),
    makePath("dir1/file1"),
    makePath("dir1/file2"),
    makePath("dir1/dir2/afile3"),
    makePath("dir1/dir2/dir3"),
    makePath("dir1/dir2/symlink_to_dir3"),
    makePath("dir1/dir2/file4"),
    makePath("dir1/dir2/dir3/file5")
};

static const fs::path RecDirFollowSymlinksIterationList[] = {
    makePath("dir1/dir2"),
    makePath("dir1/file1"),
    makePath("dir1/file2"),
    makePath("dir1/dir2/afile3"),
    makePath("dir1/dir2/dir3"),
    makePath("dir1/dir2/file4"),
    makePath("dir1/dir2/dir3/file5"),
    makePath("dir1/dir2/symlink_to_dir3"),
    makePath("dir1/dir2/symlink_to_dir3/file5"),
};

} // namespace StaticEnv

#endif // LIBCXX_FILESYSTEM_STATIC_TEST_ROOT

#ifndef LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT
#warning LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT must be defined
#else // LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT

#ifndef LIBCXX_FILESYSTEM_DYNAMIC_TEST_HELPER
#error LIBCXX_FILESYSTEM_DYNAMIC_TEST_HELPER must be defined
#endif

struct scoped_test_env
{
    scoped_test_env() : test_root(random_env_path())
        { fs_helper_run(fs_make_cmd("init_test_directory", test_root)); }

    ~scoped_test_env()
        { fs_helper_run(fs_make_cmd("destroy_test_directory", test_root)); }

    scoped_test_env(scoped_test_env const &) = delete;
    scoped_test_env & operator=(scoped_test_env const &) = delete;

    fs::path make_env_path(std::string p) { return sanitize_path(p); }

    std::string sanitize_path(std::string raw) {
        assert(raw.find("..") == std::string::npos);
        std::string const& root = test_root.native();
        if (root.compare(0, root.size(), raw, 0, root.size()) != 0) {
            assert(raw.front() != '\\');
            fs::path tmp(test_root);
            tmp /= raw;
            return std::move(const_cast<std::string&>(tmp.native()));
        }
        return raw;
    }

    std::string create_file(std::string filename, std::size_t size = 0) {
        filename = sanitize_path(std::move(filename));
        std::string out_str(size, 'a');
        {
            std::ofstream out(filename.c_str());
            out << out_str;
        }
        return filename;
    }

    std::string create_dir(std::string filename) {
        filename = sanitize_path(std::move(filename));
        fs_helper_run(fs_make_cmd("create_dir", filename));
        return filename;
    }

    std::string create_symlink(std::string source, std::string to) {
        source = sanitize_path(std::move(source));
        to = sanitize_path(std::move(to));
        fs_helper_run(fs_make_cmd("create_symlink", source, to));
        return to;
    }

    std::string create_hardlink(std::string source, std::string to) {
        source = sanitize_path(std::move(source));
        to = sanitize_path(std::move(to));
        fs_helper_run(fs_make_cmd("create_hardlink", source, to));
        return to;
    }

    std::string create_fifo(std::string file) {
        file = sanitize_path(std::move(file));
        fs_helper_run(fs_make_cmd("create_fifo", file));
        return file;
    }

  // OS X and FreeBSD doesn't support socket files so we shouldn't even
  // allow tests to call this unguarded.
#if !defined(__FreeBSD__) && !defined(__APPLE__)
    std::string create_socket(std::string file) {
        file = sanitize_path(std::move(file));
        fs_helper_run(fs_make_cmd("create_socket", file));
        return file;
    }
#endif

    fs::path const test_root;

private:
    static char to_hex(int ch) {
        return ch < 10 ? static_cast<char>('0' + ch)
                       : static_cast<char>('a' + (ch - 10));
    }

    static char random_hex_char() {
        static std::mt19937 rd { std::random_device{}() };
        static std::uniform_int_distribution<int> mrand{0, 15};
        return to_hex( mrand(rd) );
    }

    static std::string unique_path_suffix() {
        std::string model = "test.%%%%%%";
        for (auto & ch :  model) {
            if (ch == '%') ch = random_hex_char();
        }
        return model;
    }

    // This could potentially introduce a filesystem race with other tests
    // running at the same time, but oh well, it's just test code.
    static inline fs::path random_env_path() {
        static const char* env_path = LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT;
        fs::path p = fs::path(env_path) / unique_path_suffix();
        assert(p.parent_path() == env_path);
        return p;
    }

    static inline std::string make_arg(std::string const& arg) {
        return "'" + arg + "'";
    }

    static inline std::string make_arg(std::size_t arg) {
        return std::to_string(arg);
    }

    template <class T>
    static inline std::string
    fs_make_cmd(std::string const& cmd_name, T const& arg) {
        return cmd_name + "(" + make_arg(arg) + ")";
    }

    template <class T, class U>
    static inline std::string
    fs_make_cmd(std::string const& cmd_name, T const& arg1, U const& arg2) {
        return cmd_name + "(" + make_arg(arg1) + ", " + make_arg(arg2) + ")";
    }

    static inline void fs_helper_run(std::string const& raw_cmd) {
        // check that the fs test root in the enviroment matches what we were
        // compiled with.
        static bool checked = checkDynamicTestRoot();
        ((void)checked);
        std::string cmd = LIBCXX_FILESYSTEM_DYNAMIC_TEST_HELPER;
        cmd += " \"" + raw_cmd + "\"";
        int ret = std::system(cmd.c_str());
        assert(ret == 0);
    }

    static bool checkDynamicTestRoot() {
        // LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT is expected not to contain symlinks.
        char* fs_root = std::getenv("LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT");
        if (!fs_root) {
            std::printf("ERROR: LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT must be a defined "
                         "environment variable when running the test.\n");
            std::abort();
        }
        if (std::string(fs_root) != LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT) {
            std::printf("ERROR: LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT enviroment variable"
                        " must have the same value as when the test was compiled.\n");
            std::printf("   Current Value:  '%s'\n", fs_root);
            std::printf("   Expected Value: '%s'\n", LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT);
            std::abort();
        }
        return true;
    }

};

#endif // LIBCXX_FILESYSTEM_DYNAMIC_TEST_ROOT

// Misc test types

#define CONCAT2(LHS, RHS) LHS##RHS
#define CONCAT(LHS, RHS) CONCAT2(LHS, RHS)
#define MKSTR(Str) {Str, CONCAT(L, Str), CONCAT(u, Str), CONCAT(U, Str)}

struct MultiStringType {
  const char* s;
  const wchar_t* w;
  const char16_t* u16;
  const char32_t* u32;

  operator const char* () const { return s; }
  operator const wchar_t* () const { return w; }
  operator const char16_t* () const { return u16; }
  operator const char32_t* () const { return u32; }
};

const MultiStringType PathList[] = {
        MKSTR(""),
        MKSTR(" "),
        MKSTR("//"),
        MKSTR("."),
        MKSTR(".."),
        MKSTR("foo"),
        MKSTR("/"),
        MKSTR("/foo"),
        MKSTR("foo/"),
        MKSTR("/foo/"),
        MKSTR("foo/bar"),
        MKSTR("/foo/bar"),
        MKSTR("//net"),
        MKSTR("//net/foo"),
        MKSTR("///foo///"),
        MKSTR("///foo///bar"),
        MKSTR("/."),
        MKSTR("./"),
        MKSTR("/.."),
        MKSTR("../"),
        MKSTR("foo/."),
        MKSTR("foo/.."),
        MKSTR("foo/./"),
        MKSTR("foo/./bar"),
        MKSTR("foo/../"),
        MKSTR("foo/../bar"),
        MKSTR("c:"),
        MKSTR("c:/"),
        MKSTR("c:foo"),
        MKSTR("c:/foo"),
        MKSTR("c:foo/"),
        MKSTR("c:/foo/"),
        MKSTR("c:/foo/bar"),
        MKSTR("prn:"),
        MKSTR("c:\\"),
        MKSTR("c:\\foo"),
        MKSTR("c:foo\\"),
        MKSTR("c:\\foo\\"),
        MKSTR("c:\\foo/"),
        MKSTR("c:/foo\\bar"),
        MKSTR("//"),
        MKSTR("/finally/we/need/one/really/really/really/really/really/really/really/long/string")
};
const unsigned PathListSize = sizeof(PathList) / sizeof(MultiStringType);

template <class Iter>
Iter IterEnd(Iter B) {
  using VT = typename std::iterator_traits<Iter>::value_type;
  for (; *B != VT{}; ++B)
    ;
  return B;
}

template <class CharT>
const CharT* StrEnd(CharT const* P) {
    return IterEnd(P);
}

template <class CharT>
std::size_t StrLen(CharT const* P) {
    return StrEnd(P) - P;
}

// Testing the allocation behavior of the code_cvt functions requires
// *knowing* that the allocation was not done by "path::__str_".
// This hack forces path to allocate enough memory.
inline void PathReserve(fs::path& p, std::size_t N) {
  auto const& native_ref = p.native();
  const_cast<std::string&>(native_ref).reserve(N);
}

template <class Iter1, class Iter2>
bool checkCollectionsEqual(
    Iter1 start1, Iter1 const end1
  , Iter2 start2, Iter2 const end2
  )
{
    while (start1 != end1 && start2 != end2) {
        if (*start1 != *start2) {
            return false;
        }
        ++start1; ++start2;
    }
    return (start1 == end1 && start2 == end2);
}


template <class Iter1, class Iter2>
bool checkCollectionsEqualBackwards(
    Iter1 const start1, Iter1 end1
  , Iter2 const start2, Iter2 end2
  )
{
    while (start1 != end1 && start2 != end2) {
        --end1; --end2;
        if (*end1 != *end2) {
            return false;
        }
    }
    return (start1 == end1 && start2 == end2);
}

// We often need to test that the error_code was cleared if no error occurs
// this function returns a error_code which is set to an error that will
// never be returned by the filesystem functions.
inline std::error_code GetTestEC() {
    return std::make_error_code(std::errc::address_family_not_supported);
}

// Provide our own Sleep routine since std::this_thread::sleep_for is not
// available in single-threaded mode.
void SleepFor(std::chrono::seconds dur) {
    using namespace std::chrono;
#if defined(_LIBCPP_HAS_NO_MONOTONIC_CLOCK)
    using Clock = system_clock;
#else
    using Clock = steady_clock;
#endif
    const auto wake_time = Clock::now() + dur;
    while (Clock::now() < wake_time)
        ;
}

#endif /* FILESYSTEM_TEST_HELPER_HPP */
