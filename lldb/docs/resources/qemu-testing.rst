Testing LLDB using QEMU
=======================

QEMU system mode emulation
--------------------------

QEMU can be used to test LLDB in an emulation environment in the absence of
actual hardware. This page describes instructions to help setup a QEMU emulation
environment for testing LLDB.

The scripts under llvm-project/lldb/scripts/lldb-test-qemu can quickly help
setup a virtual LLDB testing environment using QEMU. The scripts currently work
with Arm or AArch64, but support for other architectures can be added easily.

* **setup.sh** is used to build the Linux kernel image and QEMU system emulation executable(s) from source.
* **rootfs.sh** is used to generate Ubuntu root file system images to be used for QEMU system mode emulation.
* **run-qemu.sh** utilizes QEMU to boot a Linux kernel image with a root file system image.

Once we have booted our kernel we can run lldb-server in emulation environment.
Ubuntu Bionic/Focal x86_64 host was used to test these scripts instructions in this
document. Please update it according to your host distribution/architecture.

.. note::
  Instructions on this page and QEMU helper scripts are verified on a Ubuntu Bionic/Focal (x86_64) host. Moreover, scripts require sudo/root permissions for installing dependencies and setting up QEMU host/guest network.

Given below are some examples of common use-cases of LLDB QEMU testing
helper scripts:

Create Ubuntu root file system image for QEMU system emulation with rootfs.sh
--------------------------------------------------------------------------------

**Example:** generate Ubuntu Bionic (armhf) rootfs image of size 1 GB
::

  $ bash rootfs.sh --arch armhf --distro bionic --size 1G

**Example:** generate Ubuntu Focal (arm64) rootfs image of size 2 GB
::

  $ bash rootfs.sh --arch arm64 --distro focal --size 2G

rootfs.sh has been tested for generating Ubuntu Bionic and Focal images but they can be used to generate rootfs images of other Debian Linux distribution.

rootfs.sh defaults username of generated image to your current username on host computer.


Build QEMU or cross compile Linux kernel from source using setup.sh
-----------------------------------------------------------------------

**Example:** Build QEMU binaries and Arm/AArch64 Linux kernel image
::

$ bash setup.sh --qemu --kernel arm
$ bash setup.sh --qemu --kernel arm64

**Example:** Build Linux kernel image only
::

$ bash setup.sh --kernel arm
$ bash setup.sh --kernel arm64

**Example:** Build qemu-system-arm and qemu-system-aarch64 binaries.
::

$ bash setup.sh --qemu

**Example:** Remove qemu.git, linux.git and linux.build from working directory
::

$ bash setup.sh --clean


Run QEMU Arm or AArch64 system emulation using run-qemu.sh
----------------------------------------------------------
run-qemu.sh has following dependencies:

* Follow https://wiki.qemu.org/Documentation/Networking/NAT and set up bridge
  networking for QEMU.

* Make sure /etc/qemu-ifup script is available with executable permissions.

* QEMU binaries must be built from source using setup.sh or provided via --qemu
  commandline argument.

* Linux kernel image must be built from source using setup.sh or provided via
  --kernel commandline argument.

* linux.build and qemu.git folder must be present in current directory if
  setup.sh was used to build Linux kernel and QEMU binaries.

* --sve option will enable AArch64 SVE mode.

* --sme option will enable AArch64 SME mode (SME requires SVE, so this will also
  be enabled).

* --mte option will enable AArch64 MTE (memory tagging) mode
  (can be used on its own or in addition to --sve).


**Example:** Run QEMU Arm or AArch64 system emulation using run-qemu.sh
::

  $ sudo bash run-qemu.sh --arch arm --rootfs <path of rootfs image>
  $ sudo bash run-qemu.sh --arch arm64 --rootfs <path of rootfs image>

**Example:** Run QEMU with kernel image and qemu binary provided using commandline
::

  $ sudo bash run-qemu.sh --arch arm64 --rootfs <path of rootfs image> \
  --kernel <path of Linux kernel image> --qemu <path of QEMU binary>


Steps for running lldb-server in QEMU system emulation environment
------------------------------------------------------------------

Using Bridge Networking
***********************

* Make sure bridge networking is enabled between host machine and QEMU VM

* Find out ip address assigned to eth0 in emulation environment

* Setup ssh access between host machine and emulation environment

* Login emulation environment and install dependencies

::

  $ sudo apt install python-dev libedit-dev libncurses5-dev libexpat1-dev

* Cross compile LLDB server for AArch64 Linux: Please visit https://lldb.llvm.org/resources/build.html for instructions on how to cross compile LLDB server.

* Transfer LLDB server executable to emulation environment

::

  $ scp lldb-server username@ip-address-of-emulation-environment:/home/username

* Run lldb-server inside QEMU VM

* Try connecting to lldb-server running inside QEMU VM with selected ip:port

Without Bridge Networking
*************************

Without bridge networking you will have to forward individual ports from the VM
to the host (refer to QEMU's manuals for the specific options).

* At least one to connect to the intial ``lldb-server``.
* One more if you want to use ``lldb-server`` in ``platform mode``, and have it
  start a ``gdbserver`` instance for you.

If you are doing either of the latter 2 you should also restrict what ports
``lldb-server tries`` to use, otherwise it will randomly pick one that is almost
certainly not forwarded. An example of this is shown below.

::

  $ lldb-server platform --server --listen 0.0.0.0:54321 --gdbserver-port 49140

The result of this is that:

* ``lldb-server`` platform mode listens externally on port ``54321``.

* When it is asked to start a new gdbserver mode instance, it will use the port
  ``49140``.

Your VM configuration should have ports ``54321`` and ``49140`` forwarded for
this to work.

QEMU user mode emulation
------------------------

Serious testing of LLDB should be done using system mode emulation. The following
is presented for information only and is not a supported testing configuration
supported by the LLDB project.

However, it is possible to run the test suite against user mode QEMU if you just
want to test a specific aspect of ``lldb`` and are ok ignoring a lot of expected
failures. This method can also be adapted for simulators with a qemu-like command
line interface.

(``lldb-server`` cannot be tested using user mode QEMU because that does not
emulate the debugging system calls that ``lldb-server`` tries to make)

Change ``LLDB_TEST_USER_ARGS`` to choose the ``qemu-user`` platform and
configure it for your architecture. The example below is for AArch64 and assumes
that ``qemu-aarch64`` is installed and on your path.

If you need to override how the ``qemu-user`` platform finds the QEMU binary,
look up the rest of the platform's settings in LLDB.

::

  -DLLDB_TEST_USER_ARGS="--platform-name;qemu-user;--setting;platform.plugin.qemu-user.architecture=aarch64;--arch;aarch64"

Also set ``LLDB_TEST_COMPILER`` to something that can target the emulated
architecture. Then you should be able to run ``ninja check-lldb`` and it will
run the tests on QEMU user automatically.

You will see a number of failures compared to a normal test run. Reasons for
this can be, but are not limited to:

* QEMU's built-in debug stub acting differently and supporting different
  features to different extents, when compared to ``lldb-server``. We try to
  be compatible but LLDB is not regularly tested with QEMU user.

* Tests that spawn new processes to attach to. QEMU user only emulates a single
  process.

* Watchpoints. Either these are not emulated or behave differently to real
  hardware. Add ``--skip-category;watchpoint`` to ``-DLLDB_TEST_USER_ARGS`` to
  skip those.

* Lack of memory region information due to QEMU communicating this in the
  GDB server format which LLDB does not use.
