%extend lldb::SBPlatform {
#ifdef SWIGPYTHON
    %pythoncode %{
        is_host = property(IsHost, None, doc='''A read only property that returns a boolean value that indiciates if this platform is the host platform.''')
    %}
#endif
}
