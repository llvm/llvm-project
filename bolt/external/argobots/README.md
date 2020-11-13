			Argobots Release %VERSION%

Argobots is a lightweight, low-level threading and tasking framework.  This
release is an experimental version of Argobots that contains features related to
user-level threads, tasklets, and some schedulers.

This README file should contain enough information to get you started with
Argobots.  More information about Argobots can be found at
http://www.argobots.org.


1. Getting Started
2. Testing Argobots
3. Reporting Problems
4. Alternate Configure Options
5. Compiler Flags
6. Developer Builds


-------------------------------------------------------------------------------

1. Getting Started
==================

The following instructions take you through a sequence of steps to get the
default configuration of Argobots up and running.

(a) You will need the following prerequisites.

    - REQUIRED: This tar file argobots-%VERSION%.tar.gz

    - REQUIRED: A C compiler (gcc is sufficient)

  Also, you need to know what shell you are using since different shell has
  different command syntax.  Command "echo $SHELL" prints out the current shell
  used by your terminal program.

(b) Unpack the tar file and go to the top level directory:

    tar xzf argobots-%VERSION%.tar.gz
    cd argobots-%VERSION%

  If your tar doesn't accept the z option, use

    gunzip argobots-%VERSION%.tar.gz
    tar xf argobots-%VERSION%.tar
    cd argobots-%VERSION%

(c) Choose an installation directory, say /home/USERNAME/argobots-install,
which is assumed to be non-existent or empty.

(d) Configure Argobots specifying the installation directory:

    for csh and tcsh:

      ./configure --prefix=/home/USERNAME/argobots-install |& tee c.txt

    for bash and sh:

      ./configure --prefix=/home/USERNAME/argobots-install 2>&1 | tee c.txt

  Bourne-like shells, sh and bash, accept "2>&1 |".  Csh-like shell, csh and
  tcsh, accept "|&".  If a failure occurs, the configure command will display
  the error.  Most errors are straight-forward to follow.

(e) Build Argobots:

    for csh and tcsh:

      make |& tee m.txt

    for bash and sh:

      make 2>&1 | tee m.txt

  This step should succeed if there were no problems with the preceding step.
  Check file m.txt.  If there were problems, do a "make clean" and then run
  make again with V=1.

    make V=1 |& tee m.txt       (for csh and tcsh)

    OR

    make V=1 2>&1 | tee m.txt   (for bash and sh)

  Then go to step 3 below, for reporting the issue to the Argobots developers
  and other users.

(f) Install Argobots:

    for csh and tcsh:

      make install |& tee mi.txt

    for bash and sh:

      make install 2>&1 | tee mi.txt

  This step collects all required executables and scripts in the bin
  subdirectory of the directory specified by the prefix argument to configure.

-------------------------------------------------------------------------------

2. Testing Argobots
===================

To test Argobots, we package the Argobots test suite in the Argobots
distribution.  You can run the test suite in the test directory using:

     make check

     OR

     make testing

The distribution also includes some Argobots examples.  You can run them in the
examples directory using:

     make check

     OR

     make testing

If you run into any problems on running the test suite or examples, please
follow step 3 below for reporting them to the Argobots developers and other
users.

-------------------------------------------------------------------------------

3. Reporting Problems
=====================

If you have problems with the installation or usage of Argobots, please follow
these steps:

(a) First visit the Frequently Asked Questions (FAQ) page at
https://github.com/pmodels/argobots/wiki/FAQ
to see if the problem you are facing has a simple solution.

(b) If you cannot find an answer on the FAQ page, look through previous email
threads on the discuss@argobots.org mailing list archive
(https://lists.argobots.org/mailman/listinfo/discuss).  It is likely someone
else had a similar problem, which has already been resolved before.

(c) If neither of the above steps work, please send an email to
discuss@argobots.org.  You need to subscribe to this list
(https://lists.argobots.org/mailman/listinfo/discuss) before sending an email.

Your email should contain the following files.  ONCE AGAIN, PLEASE COMPRESS
BEFORE SENDING, AS THE FILES CAN BE LARGE.  Note that, depending on which step
the build failed, some of the files might not exist.

    argobots-%VERSION%/c.txt (generated in step 1(d) above)
    argobots-%VERSION%/m.txt (generated in step 1(e) above)
    argobots-%VERSION%/mi.txt (generated in step 1(f) above)
    argobots-%VERSION%/config.log (generated in step 1(d) above)

    DID WE MENTION? DO NOT FORGET TO COMPRESS THESE FILES!

Finally, please include the actual error you are seeing when running the
application.  If possible, please try to reproduce the error with a smaller
application or benchmark and send that along in your bug report.

(d) If you have found a bug in Argobots, we request that you report it at our
github issues page (https://github.com/pmodels/argobots/issues).  Even if you
believe you have found a bug, we recommend you sending an email to
discuss@argobots.org first.

-------------------------------------------------------------------------------

4. Alternate Configure Options
==============================

Argobots has a number of other features.  If you are exploring Argobots as part
of a development project, you might want to tweak the Argobots build with the
following configure options.  A complete list of configuration options can be
found using:

    ./configure --help

-------------------------------------------------------------------------------

5. Compiler Flags
=================

By default, Argobots automatically adds certain compiler optimizations to
CFLAGS.  The currently used optimization level is -O2.

This optimization level can be changed with the --enable-fast option passed to
configure.  For example, to build Argobots with -O3, one can simply do:

    ./configure --enable-fast=O3

Or to disable all compiler optimizations, one can do:

    ./configure --disable-fast

For more details of --enable-fast, see the output of "./configure --help".

For performance testing, we recommend the following flags:

    ./configure --enable-fast=O3,ndebug --enable-tls-model=initial-exec \
                --enable-affinity --disable-checks

    OR

    ./configure --enable-perf-opt --enable-affinity --disable-checks

-------------------------------------------------------------------------------

6. Developer Builds
===================

For Argobots developers who want to directly work on the primary version control
system, there are a few additional steps involved (people using the release
tarballs do not have to follow these steps).  Details about these steps can be
found here: https://github.com/pmodels/argobots/wiki/Getting-and-Building

