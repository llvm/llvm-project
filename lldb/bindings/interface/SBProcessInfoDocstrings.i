%feature("docstring",
"Describes an existing process and any discoverable information that pertains to
that process."
) lldb::SBProcessInfo;

%feature("docstring",
"Return the target triple (arch-vendor-os) for the described process."
) lldb::SBProcessInfo::GetTriple;
