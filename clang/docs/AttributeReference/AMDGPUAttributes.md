## AMD GPU Attributes



### amdgpu_av

{clang-attr-syntaxes}`AMDGPUAvailableVisibleDocs`

This attribute controls availability and visibility as described in the [AMDGPU
Memory Model](https://llvm.org/docs/AMDGPUMemoryModel.html). When placed on
an atomic expression or fence, the resulting atomic or fence instruction carries
the corresponding *AV Metadata*.

The attribute takes a string literal as an argument, which currently has only
one supported value:

- `"none"`: Disable MakeAvailable and MakeVisible semantics on release and
  acquire operations respectively.

```c++
[[clang::amdgpu_av("none")]] __atomic_thread_fence(__ATOMIC_SEQ_CST);
[[clang::amdgpu_av("none")]] __atomic_fetch_add(ptr, 1, __ATOMIC_ACQ_REL);

// Also works with _Atomic type qualifier operations.
_Atomic int *p;
[[clang::amdgpu_av("none")]] *p += 1;
```


### amdgpu_flat_work_group_size

{clang-attr-syntaxes}`AMDGPUFlatWorkGroupSizeDocs`

The flat work-group size is the number of work-items in the work-group size
specified when the kernel is dispatched. It is the product of the sizes of the
x, y, and z dimension of the work-group.

Clang supports the
`__attribute__((amdgpu_flat_work_group_size(<min>, <max>)))` attribute for the
AMDGPU target. This attribute may be attached to a kernel function definition
and is an optimization hint.

`<min>` parameter specifies the minimum flat work-group size, and `<max>`
parameter specifies the maximum flat work-group size (must be greater than
`<min>`) to which all dispatches of the kernel will conform. Passing `0, 0`
as `<min>, <max>` implies the default behavior (`128, 256`).

If specified, the AMDGPU target backend might be able to produce better machine
code for barriers and perform scratch promotion by estimating available group
segment size.

An error will be given if:
: - Specified values violate subtarget specifications;
  - Specified values are not compatible with values provided through other
    attributes.


### amdgpu_max_num_work_groups

{clang-attr-syntaxes}`AMDGPUMaxNumWorkGroupsDocs`

This attribute specifies the max number of work groups when the kernel
is dispatched.

Clang supports the
`__attribute__((amdgpu_max_num_work_groups(<x>, <y>, <z>)))` or
`[[clang::amdgpu_max_num_work_groups(<x>, <y>, <z>)]]` attribute for the
AMDGPU target. This attribute may be attached to HIP or OpenCL kernel function
definitions and is an optimization hint.

The `<x>` parameter specifies the maximum number of work groups in the x dimension.
Similarly `<y>` and `<z>` are for the y and z dimensions respectively.
Each of the three values must be greater than 0 when provided. The `<x>` parameter
is required, while `<y>` and `<z>` are optional with default value of 1.

If specified, the AMDGPU target backend might be able to produce better machine
code.

An error will be given if:
: - Specified values violate subtarget specifications;
  - Specified values are not compatible with values provided through other
    attributes.


### amdgpu_num_sgpr, amdgpu_num_vgpr

{clang-attr-syntaxes}`AMDGPUNumSGPRNumVGPRDocs`

:::{warning}
These attributes are deprecated. Use the `amdgpu_waves_per_eu` attribute to
control SGPR and VGPR usage instead.
:::

Clang supports the `__attribute__((amdgpu_num_sgpr(<num_sgpr>)))` and
`__attribute__((amdgpu_num_vgpr(<num_vgpr>)))` attributes for the AMDGPU
target. These attributes may be attached to a kernel function definition and are
an optimization hint.

If these attributes are specified, then the AMDGPU target backend will attempt
to limit the number of SGPRs and/or VGPRs used to the specified value(s). The
number of used SGPRs and/or VGPRs may further be rounded up to satisfy the
allocation requirements or constraints of the subtarget. Passing `0` as
`num_sgpr` and/or `num_vgpr` implies the default behavior (no limits).

These attributes can be used to test the AMDGPU target backend. It is
recommended that the `amdgpu_waves_per_eu` attribute be used to control
resources such as SGPRs and VGPRs since it is aware of the limits for different
subtargets.

An error will be given if:
: - Specified values violate subtarget specifications;
  - Specified values are not compatible with values provided through other
    attributes;
  - The AMDGPU target backend is unable to create machine code that can meet the
    request.


### amdgpu_waves_per_eu

{clang-attr-syntaxes}`AMDGPUWavesPerEUDocs`

A compute unit (CU) is responsible for executing the wavefronts of a work-group.
It is composed of one or more execution units (EU), which are responsible for
executing the wavefronts. An EU can have enough resources to maintain the state
of more than one executing wavefront. This allows an EU to hide latency by
switching between wavefronts in a similar way to symmetric multithreading on a
CPU. In order to allow the state for multiple wavefronts to fit on an EU, the
resources used by a single wavefront have to be limited. For example, the number
of SGPRs and VGPRs. Limiting such resources can allow greater latency hiding,
but can result in having to spill some register state to memory.

Clang supports the `__attribute__((amdgpu_waves_per_eu(<min>[, <max>])))`
attribute for the AMDGPU target. This attribute may be attached to a kernel
function definition and is an optimization hint.

`<min>` parameter specifies the requested minimum number of waves per EU, and
*optional* `<max>` parameter specifies the requested maximum number of waves
per EU (must be greater than `<min>` if specified). If `<max>` is omitted,
then there is no restriction on the maximum number of waves per EU other than
the one dictated by the hardware for which the kernel is compiled. Passing
`0, 0` as `<min>, <max>` implies the default behavior (no limits).

If specified, this attribute allows an advanced developer to tune the number of
wavefronts that are capable of fitting within the resources of an EU. The AMDGPU
target backend can use this information to limit resources, such as number of
SGPRs, number of VGPRs, size of available group and private memory segments, in
such a way that guarantees that at least `<min>` wavefronts and at most
`<max>` wavefronts are able to fit within the resources of an EU. Requesting
more wavefronts can hide memory latency but limits available registers which
can result in spilling. Requesting fewer wavefronts can help reduce cache
thrashing, but can reduce memory latency hiding.

This attribute controls the machine code generated by the AMDGPU target backend
to ensure it is capable of meeting the requested values. However, when the
kernel is executed, there may be other reasons that prevent meeting the request,
for example, there may be wavefronts from other kernels executing on the EU.

An error will be given if:
: - Specified values violate subtarget specifications;
  - Specified values are not compatible with values provided through other
    attributes;

The AMDGPU target backend will emit a warning whenever it is unable to
create machine code that meets the request.


