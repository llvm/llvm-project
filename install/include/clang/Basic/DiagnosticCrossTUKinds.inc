#ifdef CROSSTUSTART
__CROSSTUSTART = DIAG_START_CROSSTU,
#undef CROSSTUSTART
#endif

DIAG(err_ctu_error_opening, CLASS_ERROR, (unsigned)diag::Severity::Error, "error opening '%0': required by the CrossTU functionality", 0, SFINAE_SubstitutionFailure, false, true, true, false, 0)
DIAG(err_extdefmap_parsing, CLASS_ERROR, (unsigned)diag::Severity::Error, "error parsing index file: '%0' line: %1 '<USR-Length>:<USR> <File-Path>' format expected", 0, SFINAE_SubstitutionFailure, false, true, true, false, 0)
DIAG(err_multiple_def_index, CLASS_ERROR, (unsigned)diag::Severity::Error, "multiple definitions are found for the same key in index ", 0, SFINAE_SubstitutionFailure, false, true, true, false, 0)
DIAG(warn_ctu_incompat_triple, CLASS_WARNING, (unsigned)diag::Severity::Warning, "imported AST from '%0' had been generated for a different target, current: %1, imported: %2", 207, SFINAE_Suppress, false, false, true, false, 0)
