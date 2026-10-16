// COM: Test Comgr parse-isa-name() API
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack-" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:sramecc-" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack+:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack-:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack-:sramecc-" SUCCESS
// RUN: parse-isa-name "spirv64-amd-amdhsa--amdgcnspirv" SUCCESS
// RUN: parse-isa-name "spirv64-amd-amdhsa-unknown-amdgcnspirv" SUCCESS

// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx1010:xnack+" SUCCESS
// RUN: parse-isa-name "" SUCCESS

// COM: Legacy aliases remain valid inputs, while unknown processors and
// COM: placeholder targets are not supported ISAs.
// RUN: parse-isa-name "amdgcn-amd-amdhsa--tahiti" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx9999" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--r600" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--generic" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--generic-hsa" INVALID_ARGUMENT

// COM: The forward-looking "amdgpu<subarch>" arch spelling is accepted in
// COM: addition to the legacy "amdgcn" arch.
// RUN: parse-isa-name "amdgpu8.03-amd-amdhsa--gfx803" SUCCESS
// RUN: parse-isa-name "amdgpu8.01-amd-amdhsa--gfx801:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu8.01-amd-amdhsa--gfx801:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu9.08-amd-amdhsa--gfx908:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.08-amd-amdhsa--gfx908:xnack+:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--gfx900" SUCCESS
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--gfx90a" SUCCESS
// RUN: parse-isa-name "amdgpu10.10-amd-amdhsa--gfx1010:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu12.50-amd-amdhsa--gfx1250" SUCCESS

// COM: Generic targets carry a major-only subarch in the new arch field.
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx9-generic" SUCCESS
// RUN: parse-isa-name "amdgpu9.4-amd-amdhsa--gfx9-4-generic" SUCCESS
// RUN: parse-isa-name "amdgpu10.1-amd-amdhsa--gfx10-1-generic" SUCCESS
// RUN: parse-isa-name "amdgpu12-amd-amdhsa--gfx12-generic" SUCCESS

// COM: A major-family subarch accepts any specific member processor it covers.
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx900" SUCCESS
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx906:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx90c" SUCCESS
// RUN: parse-isa-name "amdgpu10.3-amd-amdhsa--gfx1030" SUCCESS
// RUN: parse-isa-name "amdgpu11-amd-amdhsa--gfx1100" SUCCESS
// RUN: parse-isa-name "amdgpu12-amd-amdhsa--gfx1201" SUCCESS

// COM: gfx908 / gfx90a are their own major subarches, not covered by amdgpu9.
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx908" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx90a" INVALID_ARGUMENT
// COM: gfx1200 is family 12, not covered by the amdgpu11 major subarch.
// RUN: parse-isa-name "amdgpu11-amd-amdhsa--gfx1200" INVALID_ARGUMENT

// COM: Feature validity is keyed on the resolved processor, so xnack/sramecc
// COM: combinations behave the same regardless of the arch spelling, including
// COM: via a major-family or generic arch.
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--gfx90a:xnack+:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--gfx90a:xnack-:sramecc-" SUCCESS
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--gfx90a:xnack+:sramecc-" SUCCESS
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--gfx90a:xnack-:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.42-amd-amdhsa--gfx942:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.42-amd-amdhsa--gfx942:sramecc-" SUCCESS
// RUN: parse-isa-name "amdgpu9.50-amd-amdhsa--gfx950:xnack+:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgpu9.06-amd-amdhsa--gfx906:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx900:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu9.4-amd-amdhsa--gfx9-4-generic:xnack+:sramecc+" SUCCESS

// COM: Gfx12.5 targets support explicit xnack on/off settings.
// RUN: parse-isa-name "amdgpu12.50-amd-amdhsa--gfx1250:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu12.50-amd-amdhsa--gfx1250:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu12.50s-amd-amdhsa--gfx1250-strict:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu12.50s-amd-amdhsa--gfx1250-strict:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu12.51-amd-amdhsa--gfx1251:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu12.51-amd-amdhsa--gfx1251:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu12.5-amd-amdhsa--gfx12-5-generic:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu12.5-amd-amdhsa--gfx12-5-generic:xnack-" SUCCESS

// COM: gfx900 supports xnack but not sramecc, so a sramecc feature is rejected
// COM: regardless of the arch spelling; gfx801 supports neither sramecc nor a
// COM: xnack+sramecc combo.
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--gfx900:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--gfx900:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx900:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.01-amd-amdhsa--gfx801:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.01-amd-amdhsa--gfx801:xnack+:sramecc+" INVALID_ARGUMENT

// COM: Targets without xnack support reject any xnack feature (either polarity),
// COM: across early, RDNA, and current generations, under the new arch scheme.
// RUN: parse-isa-name "amdgpu6.00-amd-amdhsa--gfx600:xnack+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu6.00-amd-amdhsa--gfx600:xnack-" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.02-amd-amdhsa--gfx802:xnack+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu10.30-amd-amdhsa--gfx1030:xnack+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu11.00-amd-amdhsa--gfx1100:xnack+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu12.00-amd-amdhsa--gfx1200:xnack-" INVALID_ARGUMENT

// COM: The processor may be omitted; it is derived from the subarch.
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--" SUCCESS
// RUN: parse-isa-name "amdgpu9.0a-amd-amdhsa--" SUCCESS
// RUN: parse-isa-name "amdgpu12.50-amd-amdhsa--" SUCCESS
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--:xnack+" SUCCESS

// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack+:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803:::" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa-gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgcn-amd-amdhsa--gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgcn-amd-amdhsa--gfx803 " INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803 " INVALID_ARGUMENT
// RUN: parse-isa-name "   amdgcn-amd-amdhsa--gfx803  " INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803  " INVALID_ARGUMENT
// RUN: parse-isa-name "spirv64-amd-amdhsa--amdgcnspirv:xnack+" INVALID_ARGUMENT

// COM: Malformed inputs with the new "amdgpu<subarch>" arch are still rejected:
// COM: unsupported feature combo, missing environ separator, surrounding
// COM: whitespace.
// RUN: parse-isa-name "amdgpu8.01-amd-amdhsa--gfx801:xnack+:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.03-amd-amdhsa--gfx803:::" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.03-amd-amdhsa-gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgpu8.03-amd-amdhsa--gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu8.03-amd-amdhsa--gfx803 " INVALID_ARGUMENT

// COM: The environment field may be non-empty (e.g. "llvm", "unknown"); it is
// COM: validated as part of the triple rather than matched against the ISA
// COM: table, so any environment forming a valid AMDGPU triple is accepted.
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa-llvm-gfx900" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa-llvm-gfx900" SUCCESS
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa-unknown-gfx900" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa-opencl-gfx900" SUCCESS
// COM: A processor still must be consistent with the arch's subarch, regardless
// COM: of the environment.
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa-llvm-gfx803" INVALID_ARGUMENT

// COM: A processor inconsistent with the arch's subarch is rejected.
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu9.00-amd-amdhsa--gfx90a" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu9-amd-amdhsa--gfx1010" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu12.50-amd-amdhsa--gfx1251" INVALID_ARGUMENT

// COM: An unrecognized "amdgpu<subarch>" arch is rejected, including when no
// COM: processor is appended to derive from.
// RUN: parse-isa-name "amdgpufoo-amd-amdhsa--gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu99.99-amd-amdhsa--" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--" INVALID_ARGUMENT

// COM: Bare amdgpu is an alias for amdgcn when a processor is specified,
// COM: regardless of whether LLVM's TargetID parser accepts that spelling.
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900" SUCCESS
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900:xnack+" SUCCESS
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx908:xnack+:sramecc-" SUCCESS
// RUN: parse-isa-name \
// RUN:   "amdgpu-amd-amdhsa--gfx9-4-generic:sramecc+:xnack-" SUCCESS
// RUN: parse-isa-name "amdgpu-amd-amdhsa--tahiti" SUCCESS
// RUN: parse-isa-name "amdgpu-amd-amdhsa-unknown-gfx900" SUCCESS

// COM: The alias still requires a known processor and valid feature modifiers.
// RUN: parse-isa-name "amdgpu-amd-amdhsa--" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--generic" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx9999" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name \
// RUN:   "amdgpu-amd-amdhsa--gfx900:xnack+:xnack-" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900:xnack" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900:::" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa-gfx900" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgpu-amd-amdhsa--gfx900" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgpu-amd-amdhsa--gfx900 " INVALID_ARGUMENT
