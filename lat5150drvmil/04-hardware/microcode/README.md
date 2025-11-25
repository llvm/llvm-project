# Microcode Assets

Curated microcode binaries for the Core Ultra 7 165H (Meteor Lake, CPUID
`06-AA-04`). Use these payloads to stage a known-good revision into
`/lib/firmware/intel-ucode/` before regenerating initramfs so the kernel can
load it early in the boot sequence.

## Files

| File | Revision | Size | Source |
| --- | --- | --- | --- |
| `06-aa-04_0x1c.bin` | `0x1c` | 136,192 bytes | Intel Linux microcode bundle `microcode-20240312` |

## Verifying

```bash
hexdump -n 32 -C 06-aa-04_0x1c.bin
# bytes 0x04-0x07 read `1c 00 00 00`, confirming revision 0x1C
sha256sum 06-aa-04_0x1c.bin
# Expected: 36cc5efefd2ac01a25ce3b9ce73875441578749b4ce6ae02f3d370b0efccc199
```

## Installing

1. Copy the binary into the firmware directory (requires root):

   ```bash
   sudo install -m 0644 06-aa-04_0x1c.bin /lib/firmware/intel-ucode/06-aa-04
   ```

2. Regenerate initramfs so the blob is available during early boot:

   ```bash
   sudo update-initramfs -u -k all
   ```

3. Remove `dis_ucode_ldr` (and `microcode=no` if present) from
   `/etc/default/grub` **only if** you plan to let the kernel load this blob.
   Otherwise leave the loader disabled and rely on BIOS-delivered microcode.

4. Reboot and confirm `/proc/cpuinfo` reports microcode `0x1c`.

> ⚠️ Intel CPUs will refuse to load a microcode image whose revision is lower
> than the currently active one. You must prevent newer revisions (e.g., `0x24`)
> from being applied earlier in the boot chain before this downgrade can take
> effect.

