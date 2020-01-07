%extend lldb::SBTarget {
  %nothreadallow;
  const char *lldb::SBTarget::__str__ (){
    lldb::SBStream description;
    $self->GetDescription (description, lldb::eDescriptionLevelBrief);
    return description.GetData();
  }
  %clearnothreadallow;
}

