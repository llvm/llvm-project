# dsmil_avx512_enabler

Standalone Kbuild tree for the DSMIL AVX-512 enabler module that toggles the
hidden AVX-512 path on Meteor Lake hardware. This lives inside the LAT5150
framework so we can build the module with a simple `make` instead of relying on
the separate `livecd-gen` project.

## Building

```bash
cd 01-source/drivers/dsmil_avx512_enabler
make           # builds dsmil_avx512_enabler.ko against the running kernel
```

To build and copy the module into `/lib/modules/$(uname -r)/extra` in one shot:

```bash
sudo make install
```

## Loading

After a successful build, either copy the `.ko` wherever you like or load it
directly from the build directory:

```bash
sudo insmod dsmil_avx512_enabler.ko
# Unlock P-cores via the procfs interface
echo unlock | sudo tee /proc/dsmil_avx512
```

Unload with `sudo rmmod dsmil_avx512_enabler`.
