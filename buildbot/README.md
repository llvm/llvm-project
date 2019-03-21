# Scripts for build steps on Buildbot

## Purpose

The purpose of the script is for developer to customize build command for the builder on Buildbot.

## How it works

The scripts will be run by Buildbot at corresponding build step, for example, the "compile" step will run "compile.sh". Developer can change the build command, then the builder (e.g. pull request builder) will use the changed command to do the build.

## Arguments for the scripts

Common

* -b BRANCH: the branch name to build
* -n BUILD\_NUMBER: the Buildbot build number (the build count of one builder, which will be in the url of one specific build)
* -r PR\_NUMBER: if it's a pull request build, this will be the pull request number

LIT Test (check.sh)

* -t TESTCASE: the LIT testing target, e.g. check-sycl

## Assumptions

The Buildbot worker directory structure is:

    /path/to/WORKER_ROOT/BUILDER/
        llvm.src  --> source code
        llvm.obj  --> build directory

Initial working directory of the scripts:

* dependency.sh         :  llvm.obj
* configure.sh          :  llvm.obj
* compile.sh            :  llvm.obj
* clang-tidy.sh         :  llvm.src
* check.sh              :  llvm.obj
