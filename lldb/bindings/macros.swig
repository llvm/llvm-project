%define STRING_EXTENSION_LEVEL(Class, Level)
%extend {
  std::string lldb:: ## Class ## ::__repr__(){
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

%define STRING_EXTENSION(Class)
%extend {
  std::string lldb:: ## Class ## ::__repr__(){
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
