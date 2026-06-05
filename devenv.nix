{
  pkgs,
  inputs,
  ...
}:
let
  inherit (pkgs.lib.strings) optionalString hasInfix splitString;
  inherit (pkgs.lib.lists) any;

  _build_root = "build";
  _install_root = "install";
  _active_build_dir = "${_build_root}/.active";

  _llvm_targets = default: ''-DLLVM_TARGETS_TO_BUILD="''${LLVM_TARGETS:-${default}}"'';
  _llvm_projects = default: ''-DLLVM_ENABLE_PROJECTS="''${LLVM_PROJECTS:-${default}}"'';

  _llvm_runtimes =
    enabled_runtimes:
    optionalString (
      enabled_runtimes != ""
    ) ''-DLLVM_ENABLE_RUNTIMES="''${LLVM_RUNTIMES:-${enabled_runtimes}}"'';

  _libunwind_enabled = enabled_runtimes: hasInfix "libunwind" enabled_runtimes;

  # libc++abi aborts if this defaults ON without libunwind (libcxxabi/CMakeLists.txt:63)
  _libcxxabi_unwinder_flag =
    enabled_runtimes:
    optionalString (hasInfix "libcxxabi" enabled_runtimes) ''-DLIBCXXABI_USE_LLVM_UNWINDER="${
      if _libunwind_enabled enabled_runtimes then "ON" else "OFF"
    }"'';

  _clang_include_tests =
    enabled_projects:
    let
      _split_by_semi = splitString ";" enabled_projects;
    in
    optionalString (any (f: f == "clang") _split_by_semi) "-DCLANG_INCLUDE_TESTS=ON";

  _common_cmake_flags =
    enabled_projects:
    builtins.concatStringsSep " \\\n  " [
      "-DCMAKE_C_COMPILER=clang"
      "-DCMAKE_CXX_COMPILER=clang++"
      ''-DCMAKE_BUILD_TYPE="''${BUILD_TYPE:-Debug}"''
      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_USE_SPLIT_DWARF=ON"
      "-DLLVM_OPTIMIZED_TABLEGEN=ON"
      "-DLLVM_USE_LINKER=lld"
      "-DLLVM_CCACHE_BUILD=ON"
      "-DLLVM_ENABLE_ASSERTIONS=ON"
      "-DLLVM_ENABLE_EH=ON"
      "-DLLVM_ENABLE_RTTI=ON"
      "-DLLVM_INSTALL_UTILS=ON"
      "-DLLVM_INCLUDE_TESTS=ON"
      "-DLLVM_INCLUDE_EXAMPLES=OFF"
      "-DLLVM_INCLUDE_BENCHMARKS=OFF"
      "-DLLVM_ENABLE_BINDINGS=OFF"
      "-DLLVM_ENABLE_ZSTD=OFF"
      "-DLLVM_ENABLE_LIBXML2=FORCE_ON"
      "-DLLVM_ENABLE_CURL=OFF"
      "-DLLVM_ENABLE_SPHINX=OFF"
      "-DLLVM_ENABLE_DOXYGEN=OFF"
      "-DLLVM_APPEND_VC_REV=OFF"
      "${_clang_include_tests enabled_projects}"
    ];

  _make_configure_llvm_script =
    {
      name,
      projects ? "llvm",
      enabled_runtimes ? "",
    }:
    /* bash */ ''
      jobs=$(( $(${pkgs.coreutils}/bin/nproc) / 3 ))
      [ "$jobs" -lt 1 ] && jobs=1
      build_dir="''${BUILD_DIR:-${_build_root}/${name}}"
      install_dir="''${INSTALL_DIR:-${_install_root}/${name}}"
      ${pkgs.cmake}/bin/cmake -S llvm -B "$build_dir" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        ${_llvm_targets "host"} \
        ${_llvm_projects projects} \
        ${_llvm_runtimes enabled_runtimes} \
        ${_common_cmake_flags projects} \
        -DLLVM_PARALLEL_LINK_JOBS="$jobs" \
        ${_libcxxabi_unwinder_flag enabled_runtimes} \
        "$@" || exit $?
      mkdir -p "${_build_root}"
      printf '%s\n' "$build_dir" > "${_active_build_dir}"
    '';

  _resolve_build_dir = /* bash */ ''
    build_dir="''${BUILD_DIR:-}"
    if [ -z "$build_dir" ] && [ -f "${_active_build_dir}" ]; then
      build_dir="$(cat "${_active_build_dir}")"
    fi
    if [ -z "$build_dir" ]; then
      echo "No build dir selected. Run a configure-* script first or set BUILD_DIR." >&2
      exit 1
    fi
  '';

  /*
      The following defines commands that are meant to be run at the root of
      the project.
      We define configurations for `llvm` alone plus common enabled projects
      and runtimes. It creates a build directory in the root specific for
      that combination of enabled projects + runtimes.
      For example, running `configure-clang-lld` script configures a build for
      `llvm` plus the `clang` and `lld` subprojects and no enabled runtimes.
      This will create a `build/clang` repo and also create `build/.active`
      file if it doesn't exist. This file will have the currently building
      dir written in there to be referred to by the various
      `build/install/clean` commands.
  */
  _commands = [
    {
      name = "configure";
      category = "configure";
      help = "clang + lld (same as configure-clang-lld)";
      body = _make_configure_llvm_script {
        name = "clang-lld";
        projects = "clang;lld";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-llvm";
      category = "configure";
      help = "LLVM only, no projects or runtimes";
      body = _make_configure_llvm_script {
        name = "llvm";
        projects = "llvm";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-llvm-with-libcxx";
      category = "configure";
      help = "LLVM only + libc++/libc++abi runtimes";
      body = _make_configure_llvm_script {
        name = "llvm-with-libcxx";
        projects = "llvm";
        enabled_runtimes = "libcxx;libcxxabi";
      };
    }
    {
      name = "configure-llvm-with-libunwind";
      category = "configure";
      help = "LLVM only + libc++/libc++abi/libunwind runtimes";
      body = _make_configure_llvm_script {
        name = "llvm-with-libunwind";
        projects = "llvm";
        enabled_runtimes = "libcxx;libcxxabi;libunwind";
      };
    }
    {
      name = "configure-clang";
      category = "configure";
      help = "clang only";
      body = _make_configure_llvm_script {
        name = "clang";
        projects = "clang";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-clang-lld";
      category = "configure";
      help = "clang + lld";
      body = _make_configure_llvm_script {
        name = "clang-lld";
        projects = "clang;lld";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-clang-tools";
      category = "configure";
      help = "clang + clang-tools-extra + lld";
      body = _make_configure_llvm_script {
        name = "clang-tools";
        projects = "clang;clang-tools-extra;lld";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-clang-with-libcxx";
      category = "configure";
      help = "clang + lld against libc++ with compiler-rt";
      body = _make_configure_llvm_script {
        name = "clang-with-libcxx";
        projects = "clang;lld";
        enabled_runtimes = "compiler-rt;libcxxabi;libcxx";
      };
    }
    {
      name = "configure-mlir";
      category = "configure";
      help = "MLIR + clang (clang gives lit tests something to compile with)";
      body = _make_configure_llvm_script {
        name = "mlir";
        projects = "mlir;clang";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-flang";
      category = "configure";
      help = "Fortran (CMake adds mlir, clang and the flang-rt runtime)";
      body = _make_configure_llvm_script {
        name = "flang";
        projects = "flang";
        enabled_runtimes = "";
      };
    }
    {
      name = "configure-all";
      category = "configure";
      help = "every side-by-side project (bolt, clang, lld, lldb, mlir, polly, ...)";
      body = _make_configure_llvm_script {
        name = "all";
        projects = "all";
        enabled_runtimes = "";
      };
    }
    {
      name = "build";
      category = "build";
      help = "cmake --build the last configured variant";
      body = /* bash */ ''
        ${_resolve_build_dir}
        ${pkgs.cmake}/bin/cmake --build "$build_dir" "$@"
      '';
    }
    {
      name = "install";
      category = "build";
      help = "cmake --install the active build dir";
      body = /* bash */ ''
        ${_resolve_build_dir}
        ${pkgs.cmake}/bin/cmake --install "$build_dir"
      '';
    }
    {
      name = "clean-all";
      category = "build";
      help = "remove build/ and install/";
      body = /* bash */ ''
        rm -rf "${_build_root}"
      '';
    }
    {
      name = "clean";
      category = "build";
      help = "remove build/ and install/";
      body = /* bash */ ''
        ${_resolve_build_dir}
        rm -rf "$build_dir"
      '';
    }
    {
      name = "gcc-fix-compiledb";
      category = "misc";
      help = "rewrite compile_commands.json paths for host use";
      body = /* bash */ ''
        ${_resolve_build_dir}
        SRC="$(pwd)"
        BUILD="$SRC/$build_dir"
        sed -e "s|/src/|$SRC/|g" -e "s|/build/|$BUILD/|g" "$BUILD/compile_commands.json" > "$SRC/compile_commands.json"
        echo "Fixed compile_commands.json written to $SRC/compile_commands.json"
      '';
    }
  ];

  _devshell_pkg =
    (inputs.devshell.legacyPackages.${pkgs.stdenv.hostPlatform.system}.eval {
      configuration = {
        name = "llvm-dev";
        commands = map (c: {
          inherit (c) name category help;
          command = c.body;
        }) _commands;
      };
    }).config.devshell.package;

  _scripts = builtins.listToAttrs (
    map (c: {
      name = c.name;
      value = {
        exec = c.body;
        description = c.help;
      };
    }) _commands
  );
in
{
  packages = with pkgs; [
    bash
    ccache
    clang-tools
    cmake
    coreutils
    git
    gnumake
    libxml2
    lld
    ninja
    python314
    ripgrep
  ];

  languages.c.enable = true;
  languages.cplusplus.enable = true;

  env.CC = "clang";
  env.CXX = "clang++";
  # NixOS has no /usr/lib; prebuilt toolchains need libstdc++ from here.
  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

  scripts = _scripts // {
    menu = {
      exec = "${_devshell_pkg}/bin/menu";
      description = "print this command menu";
    };
  };

  enterShell = /* bash */ ''
    ${_devshell_pkg}/bin/menu
  '';
}
