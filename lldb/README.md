# Swift Debugger and REPL

**Welcome to the Swift Debugger and REPL!**

Swift is a new, high performance systems programming language.  It has a clean
and modern syntax, and offers seamless access to existing C and Objective-C
code and frameworks, and is memory safe (by default).

This repository covers the Swift Debugger and REPL support, built on
top of the LLDB Debugger.

# Building LLDB for Swift

To build LLDB for Swift, you must have the following pre-requisites
installed on your development system:

* OS X Requirements

  * OS X 10.11 (El Capitan) or later.

  * Xcode 7.1 or later.

  * [Cmake][cmake] (version 2.8.12.2 or later)

* Linux x86_64 (Ubuntu 14.04, Ubuntu 15.10, RHEL 7)

  * clang 3.5 or later.

  * [Cmake][cmake] (version 2.8.12.2 or later)

  * python 2.7

Once the pre-requisites are satisfied, follow these steps from a
bash-like shell:

```
mkdir myswift
cd myswift
git clone https://github.com/apple/swift-lldb.git lldb
lldb/scripts/build-swift-cmake.py --test
```

The lldb build script will clone additional repositories for required
dependencies if they are not already present. An optional `--update`
argument can be used to refresh these required repositories. Products
of the build process will be placed in the `build/` directory
under the root source directory.

# Inter-project Directory Layout

LLDB for Swift introduces new dependencies that do not exist with
core LLDB. In particular, LLDB for Swift makes extensive use of the
Swift codebase.

Each one of directories listed below underneath the overall
source_root are backed by a Swift.org repository:

```
.
+-- clang/
|
+-- cmark/
|
+-- lldb/
|
+-- llvm/
|
+-- ninja/
|
+-- swift/
```

Details on the contents:

* clang

  contains the stable version of clang used by Swift.

* cmark

  contains markdown support used by Swift.

* lldb

  Contains the LLDB source that includes Swift support. All of
  LLDB for Swift is contained in this repository. Core LLDB contents
  are merged into this repository. No other copy of LLDB source code
  is required.

* llvm

  Contains the stable version of llvm used by Swift.

* ninja

  Contains the ninja build system.

* swift

  Contains the Swift Language and Swift Runtime code.


# Contribution Subtleties

The swift-lldb project enhances the core LLDB project developed under
the [LLVM Project][llvm]. Swift support in the debugger is added via
the existing source-level plugin infrastructure, isolated to files that
are newly introduced in the lldb-swift repository.

Files that come from the [core LLDB project][lldb] can be readily
identified by their use of the LLVM comment header.  As no local
changes should be made to any of these files, follow the standard
[guidance for upstream changes][upstream].

[llvm]: http://llvm.org "The LLVM Project"
[lldb]: http://lldb.llvm.org "LLDB debugger"
[cmake]: https://cmake.org
