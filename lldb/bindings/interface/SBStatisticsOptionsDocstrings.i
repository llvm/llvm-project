%feature("docstring",
"A container for options to use when dumping statistics."
) lldb::SBStatisticsOptions;

%feature("docstring", "Sets whether the statistics should only dump a summary."
) lldb::SBStatisticsOptions::SetSummaryOnly;
%feature("docstring", "Gets whether the statistics only dump a summary."
) lldb::SBStatisticsOptions::GetSummaryOnly;
