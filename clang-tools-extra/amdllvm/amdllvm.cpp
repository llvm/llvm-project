#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <linux/limits.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static bool AMDCLANG_DEBUG = false;

static std::string get_path_to_self() {
    std::string p;
    std::vector<char> v(PATH_MAX);
    for (;;) {
        auto len = readlink("/proc/self/exe", v.data(), v.size());
        if (len == -1) {
            std::cerr << "amdclang: unknown readlink error" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if ((size_t)len < v.size()) {
          p = std::string(v.data());
          break;
        }
        // the path from readlink is probably truncated
        // increase the buffer size and try again
        if (v.size() < PATH_MAX * 4) {
            v.resize(v.size() * 2);
        } else {
            std::cerr << "amdclang: path from readlink exceeds the limit" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    if (AMDCLANG_DEBUG)
        std::cout << "amdclang: path to self: " << p << std::endl;
    return p;
}

static std::string get_compiler_tool(const char* command) {
    // extract the name of the command being invoked
    std::filesystem::path c{command};
    std::string filename = c.filename();
    const std::string amd{"amd"};
    if (filename.length() <= amd.length() || filename.find_first_of(amd) != 0) {
        std::cerr << "amdclang: unsupported command '" << filename << "'" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    filename = filename.substr(amd.length());

    // this provides the absolute path to this tool
    std::filesystem::path compiler_tool_path(get_path_to_self());
    compiler_tool_path.replace_filename(filename);
    if (!std::filesystem::exists(compiler_tool_path)) {
        std::cerr << "amdclang: '" << compiler_tool_path << "' doesn't exist" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // the generated compiler_tool_path could be a symlink
    // check if the symlink is pointing to a regular file
    auto wc_ctp = std::filesystem::weakly_canonical(compiler_tool_path);
    if (!std::filesystem::is_regular_file(wc_ctp)) {
        std::cerr << "amdclang: '" << compiler_tool_path 
            << "' target not a regular file" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (AMDCLANG_DEBUG)
        std::cout << "amdclang: compiler tool: " << compiler_tool_path << std::endl;

    return compiler_tool_path;
}

int main(int argc, char* argv[], char *const envp[]) {
  char* debug_env_var = getenv("AMDCLANG_DEBUG");
  if (debug_env_var != nullptr)
      AMDCLANG_DEBUG = debug_env_var[0] == '1';
  std::string compiler_tool = get_compiler_tool(argv[0]);
  pid_t pid;
  pid = vfork();
  if (pid == 0) {
      std::vector<char*> new_argv(argc + 1);
      std::vector<char> new_argv0(compiler_tool.length() + 1);
      std::memcpy(new_argv0.data(), compiler_tool.c_str(), new_argv0.size());
      new_argv[0] = new_argv0.data();
      for (int i = 1; i < argc; ++i)
        new_argv[i] = argv[i];
      new_argv[argc] = nullptr;
      // launch the compiler tool in the child process
      execve(compiler_tool.c_str(), new_argv.data(), envp);
  } else {
      int status = 0;
      pid_t wpid = waitpid(pid, &status, 0);
      if (wpid == -1) {
          std::cerr << "amdclang: child process returns an error" << std::endl;
          std::exit(EXIT_FAILURE);
      }
      if (WIFEXITED(status)) {
          // return the exit status from child
          std::exit(WEXITSTATUS(status));
      }
  }
  // exiting for an unknown reason
  std::exit(EXIT_FAILURE);
}