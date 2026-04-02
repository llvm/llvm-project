STRING_EXTENSION_OUTSIDE(SBModuleSpec)

STRING_EXTENSION_OUTSIDE(SBModuleSpecList)

%extend lldb::SBModuleSpec {
    UUIDSpan GetUUIDBytes() {
        return UUIDSpan{ $self->GetUUIDBytes(), $self->GetUUIDLength() };
    }

    bool SetUUIDBytes(UUIDSpan span) { 
        return $self->SetUUIDBytes(span.data, span.length);
    }
}
// ignore the original implementation.
%ignore lldb::SBModuleSpec::GetUUIDBytes;
