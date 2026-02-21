# ScriptedBuilder Buildbot Workers

This directory contains code shared by LLVM Buildbot workers. The typical
pipeline of a ScriptedBuilder-based builder is as follows.

1. A commit is pushed to [main](https://github.com/llvm/llvm-project/tree/main)

2. The [Buildbot master](https://lab.llvm.org/) polls the repository and finds
   new commits. It schedules build requests on every relevant worker.
   Alternatively, a build request of a specific llvm-project commit can be
   created using the "Force Build" or "Rebuild" buttons.

3. When a worker is ready, the master sends the build steps determined by
   [ScriptedBuilder](https://github.com/llvm/llvm-zorg/blob/main/zorg/buildbot/builders/ScriptedBuilder.py)
   to the worker.

4. The checkout step checks out llvm-project commit into a directory named
   `llvm.src` on the worker.

5. The annotate step executes a predefined Python script from the llvm-project
   source tree on the worker. Its working directory is an initially empty
   sibling directory named `build`. The argument `--workdir=.` is passed to
   override the default build directory (which is different to avoid
   accidentally spilling clutter into the cwd)

6. The script is expected to use the utilities from
   [`worker.py`](https://github.com/llvm/llvm-project/blob/main/.ci/buildbot/worker.py)
   to build LLVM. The `with w.step("stepname"):` pattern is used to visually
   separate additional steps in the [Buildbot GUI](https://lab.llvm.org/).


## Reproducing Builds

Users can execute the worker script directly to reproduce a build problem with
a worker using the llvm-project source tree in which it is located. By default
it will use a new directory with `.workdir` suffix (so it can be
`.gitignore`-ignored) next to the script as build directory.

The ScriptedBuilder system tries to keep all worker/build settings within the
script, but some parameters can be overridden using command line parameters.
For instance, `--jobs` overrides the Ninja and llvm-lit `-j` argument that the
worker would use. Script should be written to honor these overrides where they
apply, and may also add additional ones.

See
[`worker.py`](https://github.com/llvm/llvm-project/blob/main/.ci/buildbot/worker.py)
or a
[reference script](https://github.com/llvm/llvm-project/blob/main/polly/polly-x86_64-linux-test-suite)
for further details.
