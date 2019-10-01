# Spack Usage

This directory holds a [Spack](https://spack.io)
[repo](https://spack.readthedocs.io/en/latest/repositories.html) that can be
added to an existing Spack installation to enable the Kitsune Spack package.
The simplest invocation to accomplish this is:

```bash
spack repo add path/to/kitsune/ecp-tools/spack/repo
```

From there, you should be able to run e.g. `spack info kitsune`, `spack install
kitsune@develop` etc. as normal.

The default installation with Spack will enable all of Kitsune's supported
parallel runtimes, as well as support for lowering of Kokkos constructs.
Loading the generated environment module (`spack load -r kitsune`) will put the
Kitsune versions of `clang`, `clang++`, et al. in your `PATH`, but if you want
to use them with your build system, you'll still need to set `CC` or
`CMAKE_C_COMPILER` appropriately.
