%extend lldb::SBLineSpec {
#ifdef SWIGPYTHON
    %pythoncode %{
        @property
        def file(self) -> "SBFileSpec":
            """An `SBFileSpec` for the source file to search in."""
            return self.GetFileSpec()

        @file.setter
        def file(self, value: "SBFileSpec") -> None:
            self.SetFileSpec(value)

        @property
        def line(self) -> int:
            """The 1-based line number to search for."""
            return self.GetLine()

        @line.setter
        def line(self, value: int) -> None:
            self.SetLine(value)

        @property
        def column(self) -> int:
            """The 1-based column number, or LLDB_INVALID_COLUMN_NUMBER for any column."""
            return self.GetColumn()

        @column.setter
        def column(self, value: int) -> None:
            self.SetColumn(value)

        @property
        def check_inlines(self) -> bool:
            """Whether the search should include inlined instances of the source file that live in other compile units."""
            return self.GetCheckInlines()

        @check_inlines.setter
        def check_inlines(self, value: bool) -> None:
            self.SetCheckInlines(value)
    %}
#endif
}
