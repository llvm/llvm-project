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

// NOTE:
// This is for when you are peforming the extension from outside of the class
// instead of inside of it. If this change works out, it will replace
// STRING_EXTENSION entirely.
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
