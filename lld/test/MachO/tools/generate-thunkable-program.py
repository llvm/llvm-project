#!/usr/bin/env python3

"""Generate many skeletal functions with a thick call graph spanning a
large address space to induce lld to create branch-islands for arm64.

"""
from __future__ import print_function
import random
import argparse
import string
from pprint import pprint
from math import factorial
from itertools import permutations

# This list comes from libSystem.tbd and contains a sizeable subset
# of dylib calls available for all MacOS target archs.
libSystem_calls = (
  "__CurrentRuneLocale", "__DefaultRuneLocale", "__Exit", "__NSGetArgc",
  "__NSGetArgv", "__NSGetEnviron", "__NSGetMachExecuteHeader",
  "__NSGetProgname", "__PathLocale", "__Read_RuneMagi", "___Balloc_D2A",
  "___Bfree_D2A", "___ULtod_D2A", "____mb_cur_max", "____mb_cur_max_l",
  "____runetype", "____runetype_l", "____tolower", "____tolower_l",
  "____toupper", "____toupper_l", "___add_ovflpage", "___addel",
  "___any_on_D2A", "___assert_rtn", "___b2d_D2A", "___big_delete",
  "___big_insert", "___big_keydata", "___big_return", "___big_split",
  "___bigtens_D2A", "___bt_close", "___bt_cmp", "___bt_defcmp",
  "___bt_defpfx", "___bt_delete", "___bt_dleaf", "___bt_fd",
  "___bt_free", "___bt_get", "___bt_new", "___bt_open", "___bt_pgin",
  "___bt_pgout", "___bt_put", "___bt_ret", "___bt_search", "___bt_seq",
  "___bt_setcur", "___bt_split", "___bt_sync", "___buf_free",
  "___call_hash", "___cleanup", "___cmp_D2A", "___collate_equiv_match",
  "___collate_load_error", "___collate_lookup", "___collate_lookup_l",
  "___copybits_D2A", "___cxa_atexit", "___cxa_finalize",
  "___cxa_finalize_ranges", "___cxa_thread_atexit", "___d2b_D2A",
  "___dbpanic", "___decrement_D2A", "___default_hash", "___default_utx",
  "___delpair", "___diff_D2A", "___dtoa", "___expand_table",
  "___fflush", "___fgetwc", "___find_bigpair", "___find_last_page",
  "___fix_locale_grouping_str", "___fread", "___free_ovflpage",
  "___freedtoa", "___gdtoa", "___gdtoa_locks", "___get_buf",
  "___get_page", "___gethex_D2A", "___getonlyClocaleconv",
  "___hash_open", "___hdtoa", "___hexdig_D2A", "___hexdig_init_D2A",
  "___hexnan_D2A", "___hi0bits_D2A", "___hldtoa", "___i2b_D2A",
  "___ibitmap", "___increment_D2A", "___isctype", "___istype",
  "___istype_l", "___ldtoa", "___libc_init", "___lo0bits_D2A",
  "___log2", "___lshift_D2A", "___maskrune", "___maskrune_l",
  "___match_D2A", "___mb_cur_max", "___mb_sb_limit", "___memccpy_chk",
  "___memcpy_chk", "___memmove_chk", "___memset_chk", "___mult_D2A",
  "___multadd_D2A", "___nrv_alloc_D2A", "___opendir2", "___ovfl_delete",
  "___ovfl_get", "___ovfl_put", "___pow5mult_D2A", "___put_page",
  "___quorem_D2A", "___ratio_D2A", "___rec_close", "___rec_delete",
  "___rec_dleaf", "___rec_fd", "___rec_fmap", "___rec_fpipe",
  "___rec_get", "___rec_iput", "___rec_open", "___rec_put",
  "___rec_ret", "___rec_search", "___rec_seq", "___rec_sync",
  "___rec_vmap", "___rec_vpipe", "___reclaim_buf", "___rshift_D2A",
  "___rv_alloc_D2A", "___s2b_D2A", "___sF", "___sclose", "___sdidinit",
  "___set_ones_D2A", "___setonlyClocaleconv", "___sflags", "___sflush",
  "___sfp", "___sfvwrite", "___sglue", "___sinit", "___slbexpand",
  "___smakebuf", "___snprintf_chk", "___snprintf_object_size_chk",
  "___split_page", "___sprintf_chk", "___sprintf_object_size_chk",
  "___sread", "___srefill", "___srget", "___sseek", "___stack_chk_fail",
  "___stack_chk_guard", "___stderrp", "___stdinp", "___stdoutp",
  "___stpcpy_chk", "___stpncpy_chk", "___strcat_chk", "___strcp_D2A",
  "___strcpy_chk", "___strlcat_chk", "___strlcpy_chk", "___strncat_chk",
  "___strncpy_chk", "___strtodg", "___strtopdd", "___sum_D2A",
  "___svfscanf", "___swbuf", "___swhatbuf", "___swrite", "___swsetup",
  "___tens_D2A", "___tinytens_D2A", "___tolower", "___tolower_l",
  "___toupper", "___toupper_l", "___trailz_D2A", "___ulp_D2A",
  "___ungetc", "___ungetwc", "___vsnprintf_chk", "___vsprintf_chk",
  "___wcwidth", "___wcwidth_l", "__allocenvstate", "__atexit_receipt",
  "__c_locale", "__cleanup", "__closeutx", "__copyenv",
  "__cthread_init_routine", "__deallocenvstate", "__endutxent",
  "__flockfile_debug_stub", "__fseeko", "__ftello", "__fwalk",
  "__getenvp", "__getutxent", "__getutxid", "__getutxline",
  "__inet_aton_check", "__init_clock_port", "__int_to_time",
  "__libc_fork_child", "__libc_initializer", "__long_to_time",
  "__mkpath_np", "__mktemp", "__openutx", "__os_assert_log",
  "__os_assert_log_ctx", "__os_assumes_log", "__os_assumes_log_ctx",
  "__os_avoid_tail_call", "__os_crash", "__os_crash_callback",
  "__os_crash_fmt", "__os_debug_log", "__os_debug_log_error_str",
  "__putenvp", "__pututxline", "__rand48_add", "__rand48_mult",
  "__rand48_seed", "__readdir_unlocked", "__reclaim_telldir",
  "__seekdir", "__setenvp", "__setutxent", "__sigaction_nobind",
  "__sigintr", "__signal_nobind", "__sigvec_nobind", "__sread",
  "__sseek", "__subsystem_init", "__swrite", "__time32_to_time",
  "__time64_to_time", "__time_to_int", "__time_to_long",
  "__time_to_time32", "__time_to_time64", "__unsetenvp", "__utmpxname",
  "_a64l", "_abort", "_abort_report_np", "_abs", "_acl_add_flag_np",
  "_acl_add_perm", "_acl_calc_mask", "_acl_clear_flags_np",
  "_acl_clear_perms", "_acl_copy_entry", "_acl_copy_ext",
  "_acl_copy_ext_native", "_acl_copy_int", "_acl_copy_int_native",
  "_acl_create_entry", "_acl_create_entry_np", "_acl_delete_def_file",
  "_acl_delete_entry", "_acl_delete_fd_np", "_acl_delete_file_np",
  "_acl_delete_flag_np", "_acl_delete_link_np", "_acl_delete_perm",
  "_acl_dup", "_acl_free", "_acl_from_text", "_acl_get_entry",
  "_acl_get_fd", "_acl_get_fd_np", "_acl_get_file", "_acl_get_flag_np",
  "_acl_get_flagset_np", "_acl_get_link_np", "_acl_get_perm_np",
  "_acl_get_permset", "_acl_get_permset_mask_np", "_acl_get_qualifier",
  "_acl_get_tag_type", "_acl_init", "_acl_maximal_permset_mask_np",
  "_acl_set_fd", "_acl_set_fd_np", "_acl_set_file", "_acl_set_flagset_np",
  "_acl_set_link_np", "_acl_set_permset", "_acl_set_permset_mask_np",
  "_acl_set_qualifier", "_acl_set_tag_type", "_acl_size", "_acl_to_text",
  "_acl_valid", "_acl_valid_fd_np", "_acl_valid_file_np",
  "_acl_valid_link", "_addr2ascii", "_alarm", "_alphasort",
  "_arc4random", "_arc4random_addrandom", "_arc4random_buf",
  "_arc4random_stir", "_arc4random_uniform", "_ascii2addr", "_asctime",
  "_asctime_r", "_asprintf", "_asprintf_l", "_asxprintf",
  "_asxprintf_exec", "_atexit", "_atexit_b", "_atof", "_atof_l",
  "_atoi", "_atoi_l", "_atol", "_atol_l", "_atoll", "_atoll_l",
  "_backtrace", "_backtrace_from_fp", "_backtrace_image_offsets",
  "_backtrace_symbols", "_backtrace_symbols_fd", "_basename",
  "_basename_r", "_bcopy", "_brk", "_bsd_signal", "_bsearch",
  "_bsearch_b", "_btowc", "_btowc_l", "_catclose", "_catgets",
  "_catopen", "_cfgetispeed", "_cfgetospeed", "_cfmakeraw",
  "_cfsetispeed", "_cfsetospeed", "_cfsetspeed", "_cgetcap",
  "_cgetclose", "_cgetent", "_cgetfirst", "_cgetmatch", "_cgetnext",
  "_cgetnum", "_cgetset", "_cgetstr", "_cgetustr", "_chmodx_np",
  "_clearerr", "_clearerr_unlocked", "_clock", "_clock_getres",
  "_clock_gettime", "_clock_gettime_nsec_np", "_clock_port",
  "_clock_sem", "_clock_settime", "_closedir", "_compat_mode",
  "_confstr", "_copy_printf_domain", "_creat", "_crypt", "_ctermid",
  "_ctermid_r", "_ctime", "_ctime_r", "_daemon", "_daylight",
  "_dbm_clearerr", "_dbm_close", "_dbm_delete", "_dbm_dirfno",
  "_dbm_error", "_dbm_fetch", "_dbm_firstkey", "_dbm_nextkey",
  "_dbm_open", "_dbm_store", "_dbopen", "_devname", "_devname_r",
  "_difftime", "_digittoint", "_digittoint_l", "_dirfd", "_dirname",
  "_dirname_r", "_div", "_dprintf", "_dprintf_l", "_drand48",
  "_duplocale", "_dxprintf", "_dxprintf_exec", "_ecvt", "_encrypt",
  "_endttyent", "_endusershell", "_endutxent", "_endutxent_wtmp",
  "_erand48", "_err", "_err_set_exit", "_err_set_exit_b",
  "_err_set_file", "_errc", "_errx", "_execl", "_execle", "_execlp",
  "_execv", "_execvP", "_execvp", "_exit", "_f_prealloc", "_fchmodx_np",
  "_fclose", "_fcvt", "_fdopen", "_fdopendir", "_feof", "_feof_unlocked",
  "_ferror", "_ferror_unlocked", "_fflagstostr", "_fflush", "_fgetc",
  "_fgetln", "_fgetpos", "_fgetrune", "_fgets", "_fgetwc", "_fgetwc_l",
  "_fgetwln", "_fgetwln_l", "_fgetws", "_fgetws_l", "_fileno",
  "_fileno_unlocked", "_filesec_dup", "_filesec_free",
  "_filesec_get_property", "_filesec_init", "_filesec_query_property",
  "_filesec_set_property", "_filesec_unset_property", "_flockfile",
  "_fmemopen", "_fmtcheck", "_fmtmsg", "_fnmatch", "_fopen", "_fork",
  "_forkpty", "_fparseln", "_fprintf", "_fprintf_l", "_fpurge",
  "_fputc", "_fputrune", "_fputs", "_fputwc", "_fputwc_l", "_fputws",
  "_fputws_l", "_fread", "_free_printf_comp", "_free_printf_domain",
  "_freelocale", "_freopen", "_fscanf", "_fscanf_l", "_fseek",
  "_fseeko", "_fsetpos", "_fstatvfs", "_fstatx_np", "_fsync_volume_np",
  "_ftell", "_ftello", "_ftime", "_ftok", "_ftrylockfile",
  "_fts_children", "_fts_close", "_fts_open", "_fts_open_b",
  "_fts_read", "_fts_set", "_ftw", "_fungetrune", "_funlockfile",
  "_funopen", "_fwide", "_fwprintf", "_fwprintf_l", "_fwrite",
  "_fwscanf", "_fwscanf_l", "_fxprintf", "_fxprintf_exec", "_gcvt",
  "_getbsize", "_getc", "_getc_unlocked", "_getchar", "_getchar_unlocked",
  "_getcwd", "_getdate", "_getdate_err", "_getdelim", "_getdiskbyname",
  "_getenv", "_gethostid", "_gethostname", "_getipv4sourcefilter",
  "_getlastlogx", "_getlastlogxbyname", "_getline", "_getloadavg",
  "_getlogin", "_getlogin_r", "_getmntinfo", "_getmntinfo_r_np",
  "_getmode", "_getopt", "_getopt_long", "_getopt_long_only",
  "_getpagesize", "_getpass", "_getpeereid", "_getprogname", "_gets",
  "_getsourcefilter", "_getsubopt", "_gettimeofday", "_getttyent",
  "_getttynam", "_getusershell", "_getutmp", "_getutmpx", "_getutxent",
  "_getutxent_wtmp", "_getutxid", "_getutxline", "_getvfsbyname",
  "_getw", "_getwc", "_getwc_l", "_getwchar", "_getwchar_l", "_getwd",
  "_glob", "_glob_b", "_globfree", "_gmtime", "_gmtime_r", "_grantpt",
  "_hash_create", "_hash_destroy", "_hash_purge", "_hash_search",
  "_hash_stats", "_hash_traverse", "_hcreate", "_hdestroy",
  "_heapsort", "_heapsort_b", "_hsearch", "_imaxabs", "_imaxdiv",
  "_inet_addr", "_inet_aton", "_inet_lnaof", "_inet_makeaddr",
  "_inet_net_ntop", "_inet_net_pton", "_inet_neta", "_inet_netof",
  "_inet_network", "_inet_nsap_addr", "_inet_nsap_ntoa", "_inet_ntoa",
  "_inet_ntop", "_inet_ntop4", "_inet_ntop6", "_inet_pton",
  "_initstate", "_insque", "_isalnum", "_isalnum_l", "_isalpha",
  "_isalpha_l", "_isascii", "_isatty", "_isblank", "_isblank_l",
  "_iscntrl", "_iscntrl_l", "_isdigit", "_isdigit_l", "_isgraph",
  "_isgraph_l", "_ishexnumber", "_ishexnumber_l", "_isideogram",
  "_isideogram_l", "_islower", "_islower_l", "_isnumber", "_isnumber_l",
  "_isphonogram", "_isphonogram_l", "_isprint", "_isprint_l",
  "_ispunct", "_ispunct_l", "_isrune", "_isrune_l", "_isspace",
  "_isspace_l", "_isspecial", "_isspecial_l", "_isupper", "_isupper_l",
  "_iswalnum", "_iswalnum_l", "_iswalpha", "_iswalpha_l", "_iswascii",
  "_iswblank", "_iswblank_l", "_iswcntrl", "_iswcntrl_l", "_iswctype",
  "_iswctype_l", "_iswdigit", "_iswdigit_l", "_iswgraph", "_iswgraph_l",
  "_iswhexnumber", "_iswhexnumber_l", "_iswideogram", "_iswideogram_l",
  "_iswlower", "_iswlower_l", "_iswnumber", "_iswnumber_l",
  "_iswphonogram", "_iswphonogram_l", "_iswprint", "_iswprint_l",
  "_iswpunct", "_iswpunct_l", "_iswrune", "_iswrune_l", "_iswspace",
  "_iswspace_l", "_iswspecial", "_iswspecial_l", "_iswupper",
  "_iswupper_l", "_iswxdigit", "_iswxdigit_l", "_isxdigit",
  "_isxdigit_l", "_jrand48", "_kOSThermalNotificationPressureLevelName",
  "_killpg", "_l64a", "_labs", "_lchflags", "_lchmod", "_lcong48",
  "_ldiv", "_lfind", "_link_addr", "_link_ntoa", "_llabs", "_lldiv",
  "_localeconv", "_localeconv_l", "_localtime", "_localtime_r",
  "_lockf", "_login", "_login_tty", "_logout", "_logwtmp", "_lrand48",
  "_lsearch", "_lstatx_np", "_lutimes", "_mblen", "_mblen_l",
  "_mbmb", "_mbrlen", "_mbrlen_l", "_mbrrune", "_mbrtowc", "_mbrtowc_l",
  "_mbrune", "_mbsinit", "_mbsinit_l", "_mbsnrtowcs", "_mbsnrtowcs_l",
  "_mbsrtowcs", "_mbsrtowcs_l", "_mbstowcs", "_mbstowcs_l", "_mbtowc",
  "_mbtowc_l", "_memmem", "_memset_s", "_mergesort", "_mergesort_b",
  "_mkdirx_np", "_mkdtemp", "_mkdtempat_np", "_mkfifox_np",
  "_mkostemp", "_mkostemps", "_mkostempsat_np", "_mkpath_np",
  "_mkpathat_np", "_mkstemp", "_mkstemp_dprotected_np", "_mkstemps",
  "_mkstempsat_np", "_mktemp", "_mktime", "_monaddition", "_moncontrol",
  "_moncount", "_moninit", "_monitor", "_monoutput", "_monreset",
  "_monstartup", "_mpool_close", "_mpool_filter", "_mpool_get",
  "_mpool_new", "_mpool_open", "_mpool_put", "_mpool_sync", "_mrand48",
  "_nanosleep", "_new_printf_comp", "_new_printf_domain", "_newlocale",
  "_nextwctype", "_nextwctype_l", "_nftw", "_nice", "_nl_langinfo",
  "_nl_langinfo_l", "_nrand48", "_nvis", "_off32", "_off64",
  "_offtime", "_open_memstream", "_open_with_subsystem",
  "_open_wmemstream", "_opendev", "_opendir", "_openpty", "_openx_np",
  "_optarg", "_opterr", "_optind", "_optopt", "_optreset", "_pause",
  "_pclose", "_perror", "_popen", "_posix2time", "_posix_openpt",
  "_posix_spawnp", "_printf", "_printf_l", "_psignal", "_psort",
  "_psort_b", "_psort_r", "_ptsname", "_ptsname_r", "_putc",
  "_putc_unlocked", "_putchar", "_putchar_unlocked", "_putenv",
  "_puts", "_pututxline", "_putw", "_putwc", "_putwc_l", "_putwchar",
  "_putwchar_l", "_qsort", "_qsort_b", "_qsort_r", "_querylocale",
  "_radixsort", "_raise", "_rand", "_rand_r", "_random", "_rb_tree_count",
  "_rb_tree_find_node", "_rb_tree_find_node_geq", "_rb_tree_find_node_leq",
  "_rb_tree_init", "_rb_tree_insert_node", "_rb_tree_iterate",
  "_rb_tree_remove_node", "_readdir", "_readdir_r", "_readpassphrase",
  "_reallocf", "_realpath", "_recv", "_regcomp", "_regcomp_l",
  "_regerror", "_regexec", "_regfree", "_register_printf_domain_function",
  "_register_printf_domain_render_std", "_regncomp", "_regncomp_l",
  "_regnexec", "_regwcomp", "_regwcomp_l", "_regwexec", "_regwncomp",
  "_regwncomp_l", "_regwnexec", "_remove", "_remque", "_rewind",
  "_rewinddir", "_rindex", "_rpmatch", "_sbrk", "_scandir",
  "_scandir_b", "_scanf", "_scanf_l", "_seed48", "_seekdir", "_send",
  "_setbuf", "_setbuffer", "_setenv", "_sethostid", "_sethostname",
  "_setinvalidrune", "_setipv4sourcefilter", "_setkey", "_setlinebuf",
  "_setlocale", "_setlogin", "_setmode", "_setpgrp", "_setprogname",
  "_setrgid", "_setruid", "_setrunelocale", "_setsourcefilter",
  "_setstate", "_settimeofday", "_setttyent", "_setusershell",
  "_setutxent", "_setutxent_wtmp", "_setvbuf", "_sigaction",
  "_sigaddset", "_sigaltstack", "_sigblock", "_sigdelset",
  "_sigemptyset", "_sigfillset", "_sighold", "_sigignore",
  "_siginterrupt", "_sigismember", "_signal", "_sigpause", "_sigrelse",
  "_sigset", "_sigsetmask", "_sigvec", "_skip", "_sl_add", "_sl_find",
  "_sl_free", "_sl_init", "_sleep", "_snprintf", "_snprintf_l",
  "_snvis", "_sockatmark", "_sprintf", "_sprintf_l", "_sradixsort",
  "_srand", "_srand48", "_sranddev", "_srandom", "_srandomdev",
  "_sscanf", "_sscanf_l", "_stat_with_subsystem", "_statvfs",
  "_statx_np", "_stpcpy", "_stpncpy", "_strcasecmp", "_strcasecmp_l",
  "_strcasestr", "_strcasestr_l", "_strcat", "_strcoll", "_strcoll_l",
  "_strcspn", "_strdup", "_strenvisx", "_strerror", "_strerror_r",
  "_strfmon", "_strfmon_l", "_strftime", "_strftime_l", "_strmode",
  "_strncasecmp", "_strncasecmp_l", "_strncat", "_strndup", "_strnstr",
  "_strnunvis", "_strnunvisx", "_strnvis", "_strnvisx", "_strpbrk",
  "_strptime", "_strptime_l", "_strrchr", "_strsenvisx", "_strsep",
  "_strsignal", "_strsignal_r", "_strsnvis", "_strsnvisx", "_strspn",
  "_strsvis", "_strsvisx", "_strtod", "_strtod_l", "_strtof",
  "_strtof_l", "_strtofflags", "_strtoimax", "_strtoimax_l",
  "_strtok", "_strtok_r", "_strtol", "_strtol_l", "_strtold",
  "_strtold_l", "_strtoll", "_strtoll_l", "_strtonum", "_strtoq",
  "_strtoq_l", "_strtoul", "_strtoul_l", "_strtoull", "_strtoull_l",
  "_strtoumax", "_strtoumax_l", "_strtouq", "_strtouq_l", "_strunvis",
  "_strunvisx", "_strvis", "_strvisx", "_strxfrm", "_strxfrm_l",
  "_suboptarg", "_svis", "_swab", "_swprintf", "_swprintf_l",
  "_swscanf", "_swscanf_l", "_sxprintf", "_sxprintf_exec",
  "_sync_volume_np", "_sys_errlist", "_sys_nerr", "_sys_siglist",
  "_sys_signame", "_sysconf", "_sysctl", "_sysctlbyname",
  "_sysctlnametomib", "_system", "_tcdrain", "_tcflow", "_tcflush",
  "_tcgetattr", "_tcgetpgrp", "_tcgetsid", "_tcsendbreak", "_tcsetattr",
  "_tcsetpgrp", "_tdelete", "_telldir", "_tempnam", "_tfind",
  "_thread_stack_pcs", "_time", "_time2posix", "_timegm", "_timelocal",
  "_timeoff", "_times", "_timespec_get", "_timezone", "_timingsafe_bcmp",
  "_tmpfile", "_tmpnam", "_toascii", "_tolower", "_tolower_l",
  "_toupper", "_toupper_l", "_towctrans", "_towctrans_l", "_towlower",
  "_towlower_l", "_towupper", "_towupper_l", "_tre_ast_new_catenation",
  "_tre_ast_new_iter", "_tre_ast_new_literal", "_tre_ast_new_node",
  "_tre_ast_new_union", "_tre_compile", "_tre_fill_pmatch",
  "_tre_free", "_tre_mem_alloc_impl", "_tre_mem_destroy",
  "_tre_mem_new_impl", "_tre_parse", "_tre_stack_destroy",
  "_tre_stack_new", "_tre_stack_num_objects", "_tre_tnfa_run_backtrack",
  "_tre_tnfa_run_parallel", "_tsearch", "_ttyname", "_ttyname_r",
  "_ttyslot", "_twalk", "_tzname", "_tzset", "_tzsetwall", "_ualarm",
  "_ulimit", "_umaskx_np", "_uname", "_ungetc", "_ungetwc",
  "_ungetwc_l", "_unlockpt", "_unsetenv", "_unvis", "_uselocale",
  "_usleep", "_utime", "_utmpxname", "_uuid_clear", "_uuid_compare",
  "_uuid_copy", "_uuid_generate", "_uuid_generate_random",
  "_uuid_generate_time", "_uuid_is_null", "_uuid_pack", "_uuid_parse",
  "_uuid_unpack", "_uuid_unparse", "_uuid_unparse_lower",
  "_uuid_unparse_upper", "_vasprintf", "_vasprintf_l", "_vasxprintf",
  "_vasxprintf_exec", "_vdprintf", "_vdprintf_l", "_vdxprintf",
  "_vdxprintf_exec", "_verr", "_verrc", "_verrx", "_vfprintf",
  "_vfprintf_l", "_vfscanf", "_vfscanf_l", "_vfwprintf", "_vfwprintf_l",
  "_vfwscanf", "_vfwscanf_l", "_vfxprintf", "_vfxprintf_exec",
  "_vis", "_vprintf", "_vprintf_l", "_vscanf", "_vscanf_l",
  "_vsnprintf", "_vsnprintf_l", "_vsprintf", "_vsprintf_l", "_vsscanf",
  "_vsscanf_l", "_vswprintf", "_vswprintf_l", "_vswscanf",
  "_vswscanf_l", "_vsxprintf", "_vsxprintf_exec", "_vwarn", "_vwarnc",
  "_vwarnx", "_vwprintf", "_vwprintf_l", "_vwscanf", "_vwscanf_l",
  "_vxprintf", "_vxprintf_exec", "_wait", "_wait3", "_waitpid",
  "_warn", "_warnc", "_warnx", "_wcpcpy", "_wcpncpy", "_wcrtomb",
  "_wcrtomb_l", "_wcscasecmp", "_wcscasecmp_l", "_wcscat", "_wcschr",
  "_wcscmp", "_wcscoll", "_wcscoll_l", "_wcscpy", "_wcscspn",
  "_wcsdup", "_wcsftime", "_wcsftime_l", "_wcslcat", "_wcslcpy",
  "_wcslen", "_wcsncasecmp", "_wcsncasecmp_l", "_wcsncat", "_wcsncmp",
  "_wcsncpy", "_wcsnlen", "_wcsnrtombs", "_wcsnrtombs_l", "_wcspbrk",
  "_wcsrchr", "_wcsrtombs", "_wcsrtombs_l", "_wcsspn", "_wcsstr",
  "_wcstod", "_wcstod_l", "_wcstof", "_wcstof_l", "_wcstoimax",
  "_wcstoimax_l", "_wcstok", "_wcstol", "_wcstol_l", "_wcstold",
  "_wcstold_l", "_wcstoll", "_wcstoll_l", "_wcstombs", "_wcstombs_l",
  "_wcstoul", "_wcstoul_l", "_wcstoull", "_wcstoull_l", "_wcstoumax",
  "_wcstoumax_l", "_wcswidth", "_wcswidth_l", "_wcsxfrm", "_wcsxfrm_l",
  "_wctob", "_wctob_l", "_wctomb", "_wctomb_l", "_wctrans",
  "_wctrans_l", "_wctype", "_wctype_l", "_wcwidth", "_wcwidth_l",
  "_wmemchr", "_wmemcmp", "_wmemcpy", "_wmemmove", "_wmemset",
  "_wordexp", "_wordfree", "_wprintf", "_wprintf_l", "_wscanf",
  "_wscanf_l", "_wtmpxname", "_xprintf", "_xprintf_exec"
)

def print_here_head(name):
  print("""\
(tee %s.s |llvm-mc -filetype=obj -triple %s -o %s.o) <<END_OF_FILE &""" % (name, triple, name))

def print_here_tail():
  print("""\
END_OF_FILE
""")

def print_function_head(p2align, name):
  if args.os == "macos":
      print("""\
    .section __TEXT,__text,regular,pure_instructions
    .p2align %d, 0x90
    .globl _%s
_%s:""" % (p2align, name, name))
  elif args.os == "windows":
      print("""\
    .text
    .def %s;
    .scl 2;
    .type 32;
    .endef
    .globl %s
    .p2align %d
%s:""" % (name, name, p2align, name))
  elif args.os == "linux":
      print("""\
    .text
    .p2align %d
    .globl %s
%s:""" % (p2align, name, name))

def print_function(addr, size, addrs):
  name = "x%08x" % addr
  calls = random.randint(0, size>>12)
  print_here_head(name)
  print("""\
### %s size=%x calls=%x""" % (name, size, calls))
  print_function_head(4, name)
  for i in range(calls):
      print("    bl %sx%08x\n    .p2align 4" %
            ("_" if args.os == "macos" else "",
             addrs[random.randint(0, len(addrs)-1)]))
      if args.os == "macos":
        print("    bl %s\n    .p2align 4" %
              (libSystem_calls[random.randint(0, len(libSystem_calls)-1)]))
  fill = size - 4 * (calls + 1)
  assert fill > 0
  print("""\
    .fill 0x%x
    ret""" % (fill))
  print_here_tail()

def random_seed():
  """Generate a seed that can easily be passsed back in via --seed=STRING"""
  return ''.join(random.choice(string.ascii_lowercase) for i in range(10))

def generate_sizes(base, megabytes):
  total = 0
  while total < megabytes:
      size = random.randint(0x100, 0x10000) * 0x10
      yield size
      total += size

def generate_addrs(addr, sizes):
  i = 0
  while i < len(sizes):
      yield addr
      addr += sizes[i]
      i += 1

def main():
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="""\
WRITEME
""")
  parser.add_argument('--seed', type=str, default=random_seed(),
                      help='Seed the random number generator')
  parser.add_argument('--size', type=int, default=None,
                      help='Total text size to generate, in megabytes')
  parser.add_argument('--os', type=str, default="macos",
                      help='Target OS: macos, windows, or linux')
  global args
  args = parser.parse_args()
  triples = {
      "macos": "arm64-apple-macos",
      "linux": "aarch64-pc-linux",
      "windows": "aarch64-pc-windows"
  }
  global triple
  triple = triples.get(args.os)

  print("""\
### seed=%s triple=%s
""" % (args.seed, triple))

  random.seed(args.seed)

  base = 0x4010
  megabytes = (int(args.size) if args.size else 512) * 1024 * 1024
  sizes = [size for size in generate_sizes(base, megabytes)]
  addrs = [addr for addr in generate_addrs(base, sizes)]

  for i in range(len(addrs)):
      print_function(addrs[i], sizes[i], addrs)

  print_here_head("main")
  print("""\
### _x%08x
""" % (addrs[-1] + sizes[-1]))
  print_function_head(14 if args.os == "macos" else 4, "main")
  print("    ret")
  print_here_tail()
  print("wait")


if __name__ == '__main__':
  main()
