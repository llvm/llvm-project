%define STRING_EXTENSION_LEVEL_OUTSIDE(Class, Level)
%extend lldb:: ## Class ## {
  std::string __repr__(){
    lldb::SBStream stream;
    $self->GetDescription (stream, Level);
    const char *desc = stream.GetData();
    size_t desc_len = stream.GetSize();
    if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r')) {
      --desc_len;
    }
    return std::string(desc, desc_len);
  }
}
%enddef

%define STRING_EXTENSION_OUTSIDE(Class)
%extend lldb:: ## Class ## {
  std::string __repr__(){
    lldb::SBStream stream;
    $self->GetDescription (stream);
    const char *desc = stream.GetData();
    size_t desc_len = stream.GetSize();
    if (desc_len > 0 && (desc[desc_len-1] == '\n' || desc[desc_len-1] == '\r')) {
      --desc_len;
    }
    return std::string(desc, desc_len);
  }
}
%enddef


// Replace the C++ class method (char *dst, size_t dst_len) with 
// std::string (size_t len = 0). So callers do not have to guess the required length.
// These must be invoked before the class headers are included so that
// %ignore takes effect.
%define REPLACE_BUF_GETTER_WITH_STRING(Class, Method)
%ignore lldb:: ## Class ## :: ## Method ## (char *, size_t) const;
// Emit a single wrapper for the default-argument overload instead of two.
%feature("compactdefaultargs") lldb:: ## Class ## :: ## Method;
%extend lldb:: ## Class ## {
  std::string Method(size_t len = 0) const {
    (void)len;
    const auto required_len = $self->Method(nullptr, 0);
    if (required_len <= 0)
      return std::string{};

    std::string buf(required_len - 1, '\0');
    $self->Method(buf.data(), required_len);

    return buf;
  }
}
%enddef
