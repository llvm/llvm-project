// Used to test missing checker for missing localization context comments
// in precompiled headers.

#define MyLocalizedStringInPCH(key) NSLocalizedString((key), @"")

