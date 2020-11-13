# BOLT: OpenMP over Lightweight Threads

BOLT targets a high-performing OpenMP implementation, especially specialized
for fine-grain parallelism.  BOLT utilizes a lightweight threading model for
its underlying threading mechanism.  It currently adopts Argobots, a new
holistic, low-level threading and tasking runtime, in order to overcome
shortcomings of conventional OS-level threads.  The current BOLT implementation
is based on the OpenMP runtime in LLVM, and thus it can be used with
LLVM/Clang, Intel OpenMP compiler, and GCC.  More information about BOLT can be
found at http://www.bolt-omp.org.


1. Getting Started
2. Testing BOLT
3. BOLT-Specific Environmental Variables
4. Reporting Problems
5. Alternate Build Options


-------------------------------------------------------------------------------

1. Getting Started
==================

The following instructions take you through a sequence of steps to get the
default configuration of BOLT up and running.

Henceforth, VERSION indicates the version number of the release tarball.

(a) You will need the following prerequisites.

    - REQUIRED: This tar file bolt-VERSION.tar.gz

    - REQUIRED: C and C++ compilers (gcc and g++ are sufficient)

    - REQUIRED: CMake (http://www.cmake.org/download)

    - OPTIONAL: Argobots (http://www.argobots.org)
                The BOLT release tarball includes the Argobots source code, and
                thus you can build BOLT together with the built-in Argobots.
                Of course, you can use your own Argobots build instead of the
                accompanied one.  In the latter case, we assume Argobots has
                been installed in /home/USERNAME/argobots-install.

  Also, you need to know what shell you are using since different shell has
  different command syntax.  Command "echo $SHELL" prints out the current shell
  used by your terminal program.

  Note: if you obtained BOLT via github, the following commands download the
  built-in Argobots from the Argobots repository.

    git submodule init
    git submodule update

(b) Unpack the tar file and create a build directory:

    tar xzf bolt-VERSION.tar.gz
    mkdir bolt-build
    cd bolt-build

  If your tar doesn't accept the z option, use

    gunzip bolt-VERSION.tar.gz
    tar xf bolt-VERSION.tar
    mkdir bolt-build
    cd bolt-build

(c) Choose an installation directory, say /home/USERNAME/bolt-install, which is
assumed to be non-existent or empty.

(d) Configure BOLT specifying the installation directory:

  If you want to use the built-in Argobots,

    for csh and tcsh:

      cmake ../bolt-VERSION -G "Unix Makefiles" \
          -DCMAKE_INSTALL_PREFIX=/home/USERNAME/bolt-install \
          -DCMAKE_C_COMPILER=<C compiler> \
          -DCMAKE_CXX_COMPILER=<C++ compiler> \
          -DOPENMP_TEST_C_COMPILER=<C compiler for testing> \
          -DOPENMP_TEST_CXX_COMPILER=<C++ compiler for testing> \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBOMP_USE_ARGOBOTS=on \
          |& tee c.txt

    for bash and sh:

      cmake ../bolt-VERSION -G "Unix Makefiles" \
          -DCMAKE_INSTALL_PREFIX=/home/USERNAME/bolt-install \
          -DCMAKE_C_COMPILER=<C compiler> \
          -DCMAKE_CXX_COMPILER=<C++ compiler> \
          -DOPENMP_TEST_C_COMPILER=<C compiler for testing> \
          -DOPENMP_TEST_CXX_COMPILER=<C++ compiler for testing> \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBOMP_USE_ARGOBOTS=on \
          2>&1 | tee c.txt

  If you want to use your own Argobots build,

    for csh and tcsh:

      cmake ../bolt-VERSION -G "Unix Makefiles" \
          -DCMAKE_INSTALL_PREFIX=/home/USERNAME/bolt-install \
          -DCMAKE_C_COMPILER=<C compiler> \
          -DCMAKE_CXX_COMPILER=<C++ compiler> \
          -DOPENMP_TEST_C_COMPILER=<C compiler for testing> \
          -DOPENMP_TEST_CXX_COMPILER=<C++ compiler for testing> \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBOMP_USE_ARGOBOTS=on \
          -DLIBOMP_ARGOBOTS_INSTALL_DIR=/home/USERNAME/argobots-install \
          |& tee c.txt

    for bash and sh:

      cmake ../bolt-VERSION -G "Unix Makefiles" \
          -DCMAKE_INSTALL_PREFIX=/home/USERNAME/bolt-install \
          -DCMAKE_C_COMPILER=<C compiler> \
          -DCMAKE_CXX_COMPILER=<C++ compiler> \
          -DOPENMP_TEST_C_COMPILER=<C compiler for testing> \
          -DOPENMP_TEST_CXX_COMPILER=<C++ compiler for testing> \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIBOMP_USE_ARGOBOTS=on \
          -DLIBOMP_ARGOBOTS_INSTALL_DIR=/home/USERNAME/argobots-install \
          2>&1 | tee c.txt

  Bourne-like shells, sh and bash, accept "2>&1 |".  Csh-like shell, csh and
  tcsh, accept "|&".  If a failure occurs, the cmake command will display the
  error.  Most errors are straight-forward to follow.

(e) Build BOLT:

    for csh and tcsh:

      make |& tee m.txt

    for bash and sh:

      make 2>&1 | tee m.txt

  This step should succeed if there were no problems with the preceding step.
  Check file m.txt.  If there were problems, do a "make clean" and then run
  make again with V=1 and VERBOSE=1.

    make V=1 VERBOSE=1 |& tee m.txt       (for csh and tcsh)

    OR

    make V=1 VERBOSE=1 2>&1 | tee m.txt   (for bash and sh)

  Then go to step 3 below, for reporting the issue to the BOLT developers and
  other users.

(f) Install BOLT:

    for csh and tcsh:

      make install |& tee mi.txt

    for bash and sh:

      make install 2>&1 | tee mi.txt

  This step collects all required header and library files in the directory
  specified by the prefix argument to cmake.

-------------------------------------------------------------------------------

2. Testing BOLT
===============

To test BOLT, you can run the test suite.  Compilers for testing must be
specified when you run cmake.

For example, if llvm-lit is installed:

    cd bolt-build
    NUM_PARALLEL_TESTS=16
    llvm-lit runtime/test -v -j $NUM_PARALLEL_TESTS --timeout 600

If you run into any problems on running the test suite, please follow step 3
below for reporting them to the BOLT developers and other users.

-------------------------------------------------------------------------------

3. BOLT-Specific Environmental Variables
===============

BOLT reveals several environmental variables specific to BOLT.

    KMP_ABT_NUM_ESS=<int>: Set the number of execution streams which are
                           running on OS-level threads (e.g., Pthreads).
    KMP_ABT_SCHED_SLEEP=<1|0>: If it is set to 1, sleep a scheduler when the
                               associate pools are empty.
    KMP_ABT_VERBOSE=<1|0>: If it is set to 1, print all the BOLT-specific
                           parameters on runtime initialization.
    KMP_ABT_FORK_CUTOFF=<int>: Set the cut-off threshold used for a
                               divide-and-conquer thread creation.
    KMP_ABT_FORK_NUM_WAYS=<int>: Set the number of ways for a
                                 divide-and-conquer thread creation.
    KMP_ABT_SCHED_MIN_SLEEP_NSEC=<int>: Set the minimum scheduler sleep time
                                        (nanoseconds).
    KMP_ABT_SCHED_MAX_SLEEP_NSEC=<int>: Set the maximum scheduler sleep time
                                        (nanoseconds).
    KMP_ABT_SCHED_EVENT_FREQ=<int>: Set the event-checking frequency of
                                    schedulers.
    KMP_ABT_WORK_STEAL_FREQ=<int>: Set the random work stealing frequency of
                                   schedulers.

-------------------------------------------------------------------------------

4. Reporting Problems
=====================

If you have problems with the installation or usage of BOLT, please follow
these steps:

(a) First visit the Frequently Asked Questions (FAQ) page at
https://github.com/pmodels/bolt/wiki/FAQ
to see if the problem you are facing has a simple solution.

(b) If you cannot find an answer on the FAQ page, look through previous
email threads on the discuss@bolt-omp.org mailing list archive
(https://lists.bolt-omp.org/mailman/listinfo/discuss).  It is likely
someone else had a similar problem, which has already been resolved
before.

(c) If neither of the above steps work, please send an email to
discuss@bolt-omp.org.  You need to subscribe to this list
(https://lists.bolt-omp.org/mailman/listinfo/discuss) before sending
an email.

Your email should contain the following files.  ONCE AGAIN, PLEASE
COMPRESS BEFORE SENDING, AS THE FILES CAN BE LARGE.  Note that,
depending on which step the build failed, some of the files might not
exist.

    bolt-build/c.txt (generated in step 1(d) above)
    bolt-build/m.txt (generated in step 1(e) above)
    bolt-build/mi.txt (generated in step 1(f) above)

    DID WE MENTION? DO NOT FORGET TO COMPRESS THESE FILES!

Finally, please include the actual error you are seeing when running
the application.  If possible, please try to reproduce the error with
a smaller application or benchmark and send that along in your bug
report.

(d) If you have found a bug in BOLT, we request that you report it
at our github issues page (https://github.com/pmodels/bolt/issues).
Even if you believe you have found a bug, we recommend you sending an
email to discuss@bolt-omp.org first.

-------------------------------------------------------------------------------

5. Alternate Build Options
==============================

BOLT is based on the OpenMP subproject of LLVM for runtime, and thus it uses
the same build options provided in LLVM.

Please visit http://openmp.llvm.org/ for more build options.

