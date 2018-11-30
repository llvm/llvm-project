
# Swift Debugger and REPL

|| **Status** |
|---|---|
|**macOS**         |[![Build Status](https://ci.swift.org/job/oss-lldb-incremental-osx-cmake/badge/icon)](https://ci.swift.org/job/oss-lldb-incremental-osx)|
|**Ubuntu 14.04** |[![Build Status](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-14_04/badge/icon)](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-14_04)|
|**Ubuntu 16.04** |[![Build Status](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-16_04/badge/icon)](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-16_04)|
|**Ubuntu 18.04** |[![Build Status](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-18_04/badge/icon)](https://ci.swift.org/job/oss-lldb-incremental-linux-ubuntu-18_04)|

**Welcome to the Swift Debugger and REPL!**

Swift is a new, high performance systems programming language.  It has a clean
and modern syntax, offers seamless access to existing C and Objective-C
code and frameworks, and is memory safe (by default).

This repository covers the Swift Debugger and REPL support, built on
top of the LLDB Debugger.

# Building LLDB for Swift

To build LLDB for Swift, you must have the following prerequisites
installed on your development system:

* macOS Requirements

  * macOS 10.12.6 or later.

  * [Xcode 9.2][xcode-download] or later.

  * [Cmake][cmake] (version 2.8.12.2 or later)

* Linux x86_64 (Ubuntu 14.04, Ubuntu 15.10)

  * Clang 3.5 or later.

  * [Cmake][cmake] (version 2.8.12.2 or later)

  * Python 2.7

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

Each one of the directories listed below underneath the overall
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

  Contains the stable version of Clang used by Swift.

* cmark

  Contains markdown support used by Swift.

* lldb

  Contains the LLDB source that includes Swift support. All of
  LLDB for Swift is contained in this repository. Core LLDB contents
  are merged into this repository. No other copy of LLDB source code
  is required.

* llvm

  Contains the stable version of LLVM used by Swift.

* ninja

  Contains the Ninja build system.

* swift

  Contains the Swift Language and Swift Runtime code.

Note: If you don't use the build-swift-cmake.py script to do the initial
clone of the related project repositories, you'll need to manually clone
them to the names above:
* [swift-clang][swift-clang] should be cloned as `clang`
* [swift-cmark][swift-cmark] should be cloned as `cmark`
* [swift-llvm][swift-llvm] should be cloned as `llvm`

# Contribution Subtleties

The swift-lldb project enhances the core LLDB project developed under
the [LLVM Project][llvm]. Swift support in the debugger is added via
the existing source-level plugin infrastructure, isolated to files that
are newly introduced in the lldb-swift repository.

Files that come from the [core LLDB project][lldb] can be readily
identified by their use of the LLVM comment header.  As no local
changes should be made to any of these files, follow the standard
[guidance for upstream changes][upstream].

[cmake]: https://cmake.org
[lldb]: http://lldb.llvm.org "LLDB debugger"
[llvm]: http://llvm.org "The LLVM Project"
[swift-clang]: https://github.com/apple/swift-clang
[swift-cmark]: https://github.com/apple/swift-cmark
[swift-llvm]: https://github.com/apple/swift-llvm
[upstream]: http://swift.org/contributing/#llvm-and-swift "Upstream LLVM changes"
[xcode-download]: https://developer.apple.com/xcode/download/
