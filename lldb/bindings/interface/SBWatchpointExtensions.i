STRING_EXTENSION_LEVEL_OUTSIDE(SBWatchpoint, lldb::eDescriptionLevelVerbose)

%extend lldb::SBWatchpoint {
#ifdef SWIGPYTHON
    %pythoncode%{
      # operator== is a free function, which swig does not handle, so we inject
      # our own equality operator here
      def __eq__(self, other):
        return not self.__ne__(other)

      def __hex__(self):
        return self.GetWatchAddress()

      def __len__(self):
        return self.GetWatchSize()
    %}
#endif
}
