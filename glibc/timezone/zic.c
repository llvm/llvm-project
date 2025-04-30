/* Compile .zi time zone data into TZif binary files.  */

/*
** This file is in the public domain, so clarified as of
** 2006-07-17 by Arthur David Olson.
*/

#include "version.h"
#include "private.h"
#include "tzfile.h"

#include <fcntl.h>
#include <locale.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#define	ZIC_VERSION_PRE_2013 '2'
#define	ZIC_VERSION	'3'

typedef int_fast64_t	zic_t;
#define ZIC_MIN INT_FAST64_MIN
#define ZIC_MAX INT_FAST64_MAX
#define PRIdZIC PRIdFAST64
#define SCNdZIC SCNdFAST64

#ifndef ZIC_MAX_ABBR_LEN_WO_WARN
#define ZIC_MAX_ABBR_LEN_WO_WARN	6
#endif /* !defined ZIC_MAX_ABBR_LEN_WO_WARN */

#ifdef HAVE_DIRECT_H
# include <direct.h>
# include <io.h>
# undef mkdir
# define mkdir(name, mode) _mkdir(name)
#endif

#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef S_IRUSR
#define MKDIR_UMASK (S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)
#else
#define MKDIR_UMASK 0755
#endif
/* Port to native MS-Windows and to ancient UNIX.  */
#if !defined S_ISDIR && defined S_IFDIR && defined S_IFMT
# define S_ISDIR(mode) (((mode) & S_IFMT) == S_IFDIR)
#endif

#if HAVE_SYS_WAIT_H
#include <sys/wait.h>	/* for WIFEXITED and WEXITSTATUS */
#endif /* HAVE_SYS_WAIT_H */

#ifndef WIFEXITED
#define WIFEXITED(status)	(((status) & 0xff) == 0)
#endif /* !defined WIFEXITED */
#ifndef WEXITSTATUS
#define WEXITSTATUS(status)	(((status) >> 8) & 0xff)
#endif /* !defined WEXITSTATUS */

/* The maximum ptrdiff_t value, for pre-C99 platforms.  */
#ifndef PTRDIFF_MAX
static ptrdiff_t const PTRDIFF_MAX = MAXVAL(ptrdiff_t, TYPE_BIT(ptrdiff_t));
#endif

/* The minimum alignment of a type, for pre-C11 platforms.  */
#if __STDC_VERSION__ < 201112
# define _Alignof(type) offsetof(struct { char a; type b; }, b)
#endif

/* The type for line numbers.  Use PRIdMAX to format them; formerly
   there was also "#define PRIdLINENO PRIdMAX" and formats used
   PRIdLINENO, but xgettext cannot grok that.  */
typedef intmax_t lineno;

struct rule {
	const char *	r_filename;
	lineno		r_linenum;
	const char *	r_name;

	zic_t		r_loyear;	/* for example, 1986 */
	zic_t		r_hiyear;	/* for example, 1986 */
	const char *	r_yrtype;
	bool		r_lowasnum;
	bool		r_hiwasnum;

	int		r_month;	/* 0..11 */

	int		r_dycode;	/* see below */
	int		r_dayofmonth;
	int		r_wday;

	zic_t		r_tod;		/* time from midnight */
	bool		r_todisstd;	/* is r_tod standard time? */
	bool		r_todisut;	/* is r_tod UT? */
	bool		r_isdst;	/* is this daylight saving time? */
	zic_t		r_save;		/* offset from standard time */
	const char *	r_abbrvar;	/* variable part of abbreviation */

	bool		r_todo;		/* a rule to do (used in outzone) */
	zic_t		r_temp;		/* used in outzone */
};

/*
**	r_dycode		r_dayofmonth	r_wday
*/

#define DC_DOM		0	/* 1..31 */	/* unused */
#define DC_DOWGEQ	1	/* 1..31 */	/* 0..6 (Sun..Sat) */
#define DC_DOWLEQ	2	/* 1..31 */	/* 0..6 (Sun..Sat) */

struct zone {
	const char *	z_filename;
	lineno		z_linenum;

	const char *	z_name;
	zic_t		z_stdoff;
	char *		z_rule;
	const char *	z_format;
	char		z_format_specifier;

	bool		z_isdst;
	zic_t		z_save;

	struct rule *	z_rules;
	ptrdiff_t	z_nrules;

	struct rule	z_untilrule;
	zic_t		z_untiltime;
};

#if !HAVE_POSIX_DECLS
extern int	getopt(int argc, char * const argv[],
			const char * options);
extern int	link(const char * fromname, const char * toname);
extern char *	optarg;
extern int	optind;
#endif

#if ! HAVE_LINK
# define link(from, to) (errno = ENOTSUP, -1)
#endif
#if ! HAVE_SYMLINK
# define readlink(file, buf, size) (errno = ENOTSUP, -1)
# define symlink(from, to) (errno = ENOTSUP, -1)
# define S_ISLNK(m) 0
#endif
#ifndef AT_SYMLINK_FOLLOW
# define linkat(fromdir, from, todir, to, flag) \
    (itssymlink(from) ? (errno = ENOTSUP, -1) : link(from, to))
#endif

static void	addtt(zic_t starttime, int type);
static int	addtype(zic_t, char const *, bool, bool, bool);
static void	leapadd(zic_t, int, int);
static void	adjleap(void);
static void	associate(void);
static void	dolink(const char *, const char *, bool);
static char **	getfields(char * buf);
static zic_t	gethms(const char * string, const char * errstring);
static zic_t	getsave(char *, bool *);
static void	inexpires(char **, int);
static void	infile(const char * filename);
static void	inleap(char ** fields, int nfields);
static void	inlink(char ** fields, int nfields);
static void	inrule(char ** fields, int nfields);
static bool	inzcont(char ** fields, int nfields);
static bool	inzone(char ** fields, int nfields);
static bool	inzsub(char **, int, bool);
static bool	itsdir(char const *);
static bool	itssymlink(char const *);
static bool	is_alpha(char a);
static char	lowerit(char);
static void	mkdirs(char const *, bool);
static void	newabbr(const char * abbr);
static zic_t	oadd(zic_t t1, zic_t t2);
static void	outzone(const struct zone * zp, ptrdiff_t ntzones);
static zic_t	rpytime(const struct rule * rp, zic_t wantedy);
static void	rulesub(struct rule * rp,
			const char * loyearp, const char * hiyearp,
			const char * typep, const char * monthp,
			const char * dayp, const char * timep);
static zic_t	tadd(zic_t t1, zic_t t2);
static bool	yearistype(zic_t year, const char * type);

/* Bound on length of what %z can expand to.  */
enum { PERCENT_Z_LEN_BOUND = sizeof "+995959" - 1 };

/* If true, work around a bug in Qt 5.6.1 and earlier, which mishandles
   TZif files whose POSIX-TZ-style strings contain '<'; see
   QTBUG-53071 <https://bugreports.qt.io/browse/QTBUG-53071>.  This
   workaround will no longer be needed when Qt 5.6.1 and earlier are
   obsolete, say in the year 2021.  */
#ifndef WORK_AROUND_QTBUG_53071
enum { WORK_AROUND_QTBUG_53071 = true };
#endif

static int		charcnt;
static bool		errors;
static bool		warnings;
static const char *	filename;
static int		leapcnt;
static bool		leapseen;
static zic_t		leapminyear;
static zic_t		leapmaxyear;
static lineno		linenum;
static int		max_abbrvar_len = PERCENT_Z_LEN_BOUND;
static int		max_format_len;
static zic_t		max_year;
static zic_t		min_year;
static bool		noise;
static const char *	rfilename;
static lineno		rlinenum;
static const char *	progname;
static ptrdiff_t	timecnt;
static ptrdiff_t	timecnt_alloc;
static int		typecnt;

/*
** Line codes.
*/

#define LC_RULE		0
#define LC_ZONE		1
#define LC_LINK		2
#define LC_LEAP		3
#define LC_EXPIRES	4

/*
** Which fields are which on a Zone line.
*/

#define ZF_NAME		1
#define ZF_STDOFF	2
#define ZF_RULE		3
#define ZF_FORMAT	4
#define ZF_TILYEAR	5
#define ZF_TILMONTH	6
#define ZF_TILDAY	7
#define ZF_TILTIME	8
#define ZONE_MINFIELDS	5
#define ZONE_MAXFIELDS	9

/*
** Which fields are which on a Zone continuation line.
*/

#define ZFC_STDOFF	0
#define ZFC_RULE	1
#define ZFC_FORMAT	2
#define ZFC_TILYEAR	3
#define ZFC_TILMONTH	4
#define ZFC_TILDAY	5
#define ZFC_TILTIME	6
#define ZONEC_MINFIELDS	3
#define ZONEC_MAXFIELDS	7

/*
** Which files are which on a Rule line.
*/

#define RF_NAME		1
#define RF_LOYEAR	2
#define RF_HIYEAR	3
#define RF_COMMAND	4
#define RF_MONTH	5
#define RF_DAY		6
#define RF_TOD		7
#define RF_SAVE		8
#define RF_ABBRVAR	9
#define RULE_FIELDS	10

/*
** Which fields are which on a Link line.
*/

#define LF_FROM		1
#define LF_TO		2
#define LINK_FIELDS	3

/*
** Which fields are which on a Leap line.
*/

#define LP_YEAR		1
#define LP_MONTH	2
#define LP_DAY		3
#define LP_TIME		4
#define LP_CORR		5
#define LP_ROLL		6
#define LEAP_FIELDS	7

/* Expires lines are like Leap lines, except without CORR and ROLL fields.  */
#define EXPIRES_FIELDS	5

/*
** Year synonyms.
*/

#define YR_MINIMUM	0
#define YR_MAXIMUM	1
#define YR_ONLY		2

static struct rule *	rules;
static ptrdiff_t	nrules;	/* number of rules */
static ptrdiff_t	nrules_alloc;

static struct zone *	zones;
static ptrdiff_t	nzones;	/* number of zones */
static ptrdiff_t	nzones_alloc;

struct link {
	const char *	l_filename;
	lineno		l_linenum;
	const char *	l_from;
	const char *	l_to;
};

static struct link *	links;
static ptrdiff_t	nlinks;
static ptrdiff_t	nlinks_alloc;

struct lookup {
	const char *	l_word;
	const int	l_value;
};

static struct lookup const *	byword(const char * string,
					const struct lookup * lp);

static struct lookup const zi_line_codes[] = {
	{ "Rule",	LC_RULE },
	{ "Zone",	LC_ZONE },
	{ "Link",	LC_LINK },
	{ NULL,		0 }
};
static struct lookup const leap_line_codes[] = {
	{ "Leap",	LC_LEAP },
	{ "Expires",	LC_EXPIRES },
	{ NULL,		0}
};

static struct lookup const	mon_names[] = {
	{ "January",	TM_JANUARY },
	{ "February",	TM_FEBRUARY },
	{ "March",	TM_MARCH },
	{ "April",	TM_APRIL },
	{ "May",	TM_MAY },
	{ "June",	TM_JUNE },
	{ "July",	TM_JULY },
	{ "August",	TM_AUGUST },
	{ "September",	TM_SEPTEMBER },
	{ "October",	TM_OCTOBER },
	{ "November",	TM_NOVEMBER },
	{ "December",	TM_DECEMBER },
	{ NULL,		0 }
};

static struct lookup const	wday_names[] = {
	{ "Sunday",	TM_SUNDAY },
	{ "Monday",	TM_MONDAY },
	{ "Tuesday",	TM_TUESDAY },
	{ "Wednesday",	TM_WEDNESDAY },
	{ "Thursday",	TM_THURSDAY },
	{ "Friday",	TM_FRIDAY },
	{ "Saturday",	TM_SATURDAY },
	{ NULL,		0 }
};

static struct lookup const	lasts[] = {
	{ "last-Sunday",	TM_SUNDAY },
	{ "last-Monday",	TM_MONDAY },
	{ "last-Tuesday",	TM_TUESDAY },
	{ "last-Wednesday",	TM_WEDNESDAY },
	{ "last-Thursday",	TM_THURSDAY },
	{ "last-Friday",	TM_FRIDAY },
	{ "last-Saturday",	TM_SATURDAY },
	{ NULL,			0 }
};

static struct lookup const	begin_years[] = {
	{ "minimum",	YR_MINIMUM },
	{ "maximum",	YR_MAXIMUM },
	{ NULL,		0 }
};

static struct lookup const	end_years[] = {
	{ "minimum",	YR_MINIMUM },
	{ "maximum",	YR_MAXIMUM },
	{ "only",	YR_ONLY },
	{ NULL,		0 }
};

static struct lookup const	leap_types[] = {
	{ "Rolling",	true },
	{ "Stationary",	false },
	{ NULL,		0 }
};

static const int	len_months[2][MONSPERYEAR] = {
	{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
	{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

static const int	len_years[2] = {
	DAYSPERNYEAR, DAYSPERLYEAR
};

static struct attype {
	zic_t		at;
	bool		dontmerge;
	unsigned char	type;
} *			attypes;
static zic_t		utoffs[TZ_MAX_TYPES];
static char		isdsts[TZ_MAX_TYPES];
static unsigned char	desigidx[TZ_MAX_TYPES];
static bool		ttisstds[TZ_MAX_TYPES];
static bool		ttisuts[TZ_MAX_TYPES];
static char		chars[TZ_MAX_CHARS];
static zic_t		trans[TZ_MAX_LEAPS];
static zic_t		corr[TZ_MAX_LEAPS];
static char		roll[TZ_MAX_LEAPS];

/*
** Memory allocation.
*/

static _Noreturn void
memory_exhausted(const char *msg)
{
	fprintf(stderr, _("%s: Memory exhausted: %s\n"), progname, msg);
	exit(EXIT_FAILURE);
}

static ATTRIBUTE_PURE size_t
size_product(size_t nitems, size_t itemsize)
{
	if (SIZE_MAX / itemsize < nitems)
		memory_exhausted(_("size overflow"));
	return nitems * itemsize;
}

static ATTRIBUTE_PURE size_t
align_to(size_t size, size_t alignment)
{
  size_t aligned_size = size + alignment - 1;
  aligned_size -= aligned_size % alignment;
  if (aligned_size < size)
    memory_exhausted(_("alignment overflow"));
  return aligned_size;
}

#if !HAVE_STRDUP
static char *
strdup(char const *str)
{
  char *result = malloc(strlen(str) + 1);
  return result ? strcpy(result, str) : result;
}
#endif

static void *
memcheck(void *ptr)
{
	if (ptr == NULL)
		memory_exhausted(strerror(errno));
	return ptr;
}

static void * ATTRIBUTE_MALLOC
emalloc(size_t size)
{
  return memcheck(malloc(size));
}

static void *
erealloc(void *ptr, size_t size)
{
  return memcheck(realloc(ptr, size));
}

static char * ATTRIBUTE_MALLOC
ecpyalloc (char const *str)
{
  return memcheck(strdup(str));
}

static void *
growalloc(void *ptr, size_t itemsize, ptrdiff_t nitems, ptrdiff_t *nitems_alloc)
{
	if (nitems < *nitems_alloc)
		return ptr;
	else {
		ptrdiff_t nitems_max = PTRDIFF_MAX - WORK_AROUND_QTBUG_53071;
		ptrdiff_t amax = nitems_max < SIZE_MAX ? nitems_max : SIZE_MAX;
		if ((amax - 1) / 3 * 2 < *nitems_alloc)
			memory_exhausted(_("integer overflow"));
		*nitems_alloc += (*nitems_alloc >> 1) + 1;
		return erealloc(ptr, size_product(*nitems_alloc, itemsize));
	}
}

/*
** Error handling.
*/

static void
eats(char const *name, lineno num, char const *rname, lineno rnum)
{
	filename = name;
	linenum = num;
	rfilename = rname;
	rlinenum = rnum;
}

static void
eat(char const *name, lineno num)
{
	eats(name, num, NULL, -1);
}

static void ATTRIBUTE_FORMAT((printf, 1, 0))
verror(const char *const string, va_list args)
{
	/*
	** Match the format of "cc" to allow sh users to
	**	zic ... 2>&1 | error -t "*" -v
	** on BSD systems.
	*/
	if (filename)
	  fprintf(stderr, _("\"%s\", line %"PRIdMAX": "), filename, linenum);
	vfprintf(stderr, string, args);
	if (rfilename != NULL)
		fprintf(stderr, _(" (rule from \"%s\", line %"PRIdMAX")"),
			rfilename, rlinenum);
	fprintf(stderr, "\n");
}

static void ATTRIBUTE_FORMAT((printf, 1, 2))
error(const char *const string, ...)
{
	va_list args;
	va_start(args, string);
	verror(string, args);
	va_end(args);
	errors = true;
}

static void ATTRIBUTE_FORMAT((printf, 1, 2))
warning(const char *const string, ...)
{
	va_list args;
	fprintf(stderr, _("warning: "));
	va_start(args, string);
	verror(string, args);
	va_end(args);
	warnings = true;
}

static void
close_file(FILE *stream, char const *dir, char const *name)
{
  char const *e = (ferror(stream) ? _("I/O error")
		   : fclose(stream) != 0 ? strerror(errno) : NULL);
  if (e) {
    fprintf(stderr, "%s: %s%s%s%s%s\n", progname,
	    dir ? dir : "", dir ? "/" : "",
	    name ? name : "", name ? ": " : "",
	    e);
    exit(EXIT_FAILURE);
  }
}

static _Noreturn void
usage(FILE *stream, int status)
{
  fprintf(stream,
	  _("%s: usage is %s [ --version ] [ --help ] [ -v ] \\\n"
	    "\t[ -b {slim|fat} ] [ -d directory ] [ -l localtime ]"
	    " [ -L leapseconds ] \\\n"
	    "\t[ -p posixrules ] [ -r '[@lo][/@hi]' ] [ -t localtime-link ] \\\n"
	    "\t[ filename ... ]\n\n"
	    "Report bugs to %s.\n"),
	  progname, progname, REPORT_BUGS_TO);
  if (status == EXIT_SUCCESS)
    close_file(stream, NULL, NULL);
  exit(status);
}

/* Change the working directory to DIR, possibly creating DIR and its
   ancestors.  After this is done, all files are accessed with names
   relative to DIR.  */
static void
change_directory (char const *dir)
{
  if (chdir(dir) != 0) {
    int chdir_errno = errno;
    if (chdir_errno == ENOENT) {
      mkdirs(dir, false);
      chdir_errno = chdir(dir) == 0 ? 0 : errno;
    }
    if (chdir_errno != 0) {
      fprintf(stderr, _("%s: Can't chdir to %s: %s\n"),
	      progname, dir, strerror(chdir_errno));
      exit(EXIT_FAILURE);
    }
  }
}

#define TIME_T_BITS_IN_FILE 64

/* The minimum and maximum values representable in a TZif file.  */
static zic_t const min_time = MINVAL(zic_t, TIME_T_BITS_IN_FILE);
static zic_t const max_time = MAXVAL(zic_t, TIME_T_BITS_IN_FILE);

/* The minimum, and one less than the maximum, values specified by
   the -r option.  These default to MIN_TIME and MAX_TIME.  */
static zic_t lo_time = MINVAL(zic_t, TIME_T_BITS_IN_FILE);
static zic_t hi_time = MAXVAL(zic_t, TIME_T_BITS_IN_FILE);

/* The time specified by an Expires line, or negative if no such line.  */
static zic_t leapexpires = -1;

/* The time specified by an #expires comment, or negative if no such line.  */
static zic_t comment_leapexpires = -1;

/* Set the time range of the output to TIMERANGE.
   Return true if successful.  */
static bool
timerange_option(char *timerange)
{
  intmax_t lo = min_time, hi = max_time;
  char *lo_end = timerange, *hi_end;
  if (*timerange == '@') {
    errno = 0;
    lo = strtoimax (timerange + 1, &lo_end, 10);
    if (lo_end == timerange + 1 || (lo == INTMAX_MAX && errno == ERANGE))
      return false;
  }
  hi_end = lo_end;
  if (lo_end[0] == '/' && lo_end[1] == '@') {
    errno = 0;
    hi = strtoimax (lo_end + 2, &hi_end, 10);
    if (hi_end == lo_end + 2 || hi == INTMAX_MIN)
      return false;
    hi -= ! (hi == INTMAX_MAX && errno == ERANGE);
  }
  if (*hi_end || hi < lo || max_time < lo || hi < min_time)
    return false;
  lo_time = lo < min_time ? min_time : lo;
  hi_time = max_time < hi ? max_time : hi;
  return true;
}

static const char *	psxrules;
static const char *	lcltime;
static const char *	directory;
static const char *	leapsec;
static const char *	tzdefault;
static const char *	yitcommand;

/* -1 if the TZif output file should be slim, 0 if default, 1 if the
   output should be fat for backward compatibility.  Currently the
   default is fat, although this may change.  */
static int bloat;

static bool
want_bloat(void)
{
  return 0 <= bloat;
}

#ifndef ZIC_BLOAT_DEFAULT
# define ZIC_BLOAT_DEFAULT "fat"
#endif

int
main(int argc, char **argv)
{
	register int c, k;
	register ptrdiff_t i, j;
	bool timerange_given = false;

#ifdef S_IWGRP
	umask(umask(S_IWGRP | S_IWOTH) | (S_IWGRP | S_IWOTH));
#endif
#if HAVE_GETTEXT
	setlocale(LC_ALL, "");
#ifdef TZ_DOMAINDIR
	bindtextdomain(TZ_DOMAIN, TZ_DOMAINDIR);
#endif /* defined TEXTDOMAINDIR */
	textdomain(TZ_DOMAIN);
#endif /* HAVE_GETTEXT */
	progname = argv[0];
	if (TYPE_BIT(zic_t) < 64) {
		fprintf(stderr, "%s: %s\n", progname,
			_("wild compilation-time specification of zic_t"));
		return EXIT_FAILURE;
	}
	for (k = 1; k < argc; k++)
		if (strcmp(argv[k], "--version") == 0) {
			printf("zic %s%s\n", PKGVERSION, TZVERSION);
			close_file(stdout, NULL, NULL);
			return EXIT_SUCCESS;
		} else if (strcmp(argv[k], "--help") == 0) {
			usage(stdout, EXIT_SUCCESS);
		}
	while ((c = getopt(argc, argv, "b:d:l:L:p:r:st:vy:")) != EOF && c != -1)
		switch (c) {
			default:
				usage(stderr, EXIT_FAILURE);
			case 'b':
				if (strcmp(optarg, "slim") == 0) {
				  if (0 < bloat)
				    error(_("incompatible -b options"));
				  bloat = -1;
				} else if (strcmp(optarg, "fat") == 0) {
				  if (bloat < 0)
				    error(_("incompatible -b options"));
				  bloat = 1;
				} else
				  error(_("invalid option: -b '%s'"), optarg);
				break;
			case 'd':
				if (directory == NULL)
					directory = optarg;
				else {
					fprintf(stderr,
_("%s: More than one -d option specified\n"),
						progname);
					return EXIT_FAILURE;
				}
				break;
			case 'l':
				if (lcltime == NULL)
					lcltime = optarg;
				else {
					fprintf(stderr,
_("%s: More than one -l option specified\n"),
						progname);
					return EXIT_FAILURE;
				}
				break;
			case 'p':
				if (psxrules == NULL)
					psxrules = optarg;
				else {
					fprintf(stderr,
_("%s: More than one -p option specified\n"),
						progname);
					return EXIT_FAILURE;
				}
				break;
			case 't':
				if (tzdefault != NULL) {
				  fprintf(stderr,
					  _("%s: More than one -t option"
					    " specified\n"),
					  progname);
				  return EXIT_FAILURE;
				}
				tzdefault = optarg;
				break;
			case 'y':
				if (yitcommand == NULL) {
					warning(_("-y is obsolescent"));
					yitcommand = optarg;
				} else {
					fprintf(stderr,
_("%s: More than one -y option specified\n"),
						progname);
					return EXIT_FAILURE;
				}
				break;
			case 'L':
				if (leapsec == NULL)
					leapsec = optarg;
				else {
					fprintf(stderr,
_("%s: More than one -L option specified\n"),
						progname);
					return EXIT_FAILURE;
				}
				break;
			case 'v':
				noise = true;
				break;
			case 'r':
				if (timerange_given) {
				  fprintf(stderr,
_("%s: More than one -r option specified\n"),
					  progname);
				  return EXIT_FAILURE;
				}
				if (! timerange_option(optarg)) {
				  fprintf(stderr,
_("%s: invalid time range: %s\n"),
					  progname, optarg);
				  return EXIT_FAILURE;
				}
				timerange_given = true;
				break;
			case 's':
				warning(_("-s ignored"));
				break;
		}
	if (optind == argc - 1 && strcmp(argv[optind], "=") == 0)
		usage(stderr, EXIT_FAILURE);	/* usage message by request */
	if (bloat == 0)
	  bloat = strcmp(ZIC_BLOAT_DEFAULT, "slim") == 0 ? -1 : 1;
	if (directory == NULL)
		directory = TZDIR;
	if (tzdefault == NULL)
		tzdefault = TZDEFAULT;
	if (yitcommand == NULL)
		yitcommand = "yearistype";

	if (optind < argc && leapsec != NULL) {
		infile(leapsec);
		adjleap();
	}

	for (k = optind; k < argc; k++)
		infile(argv[k]);
	if (errors)
		return EXIT_FAILURE;
	associate();
	change_directory(directory);
	for (i = 0; i < nzones; i = j) {
		/*
		** Find the next non-continuation zone entry.
		*/
		for (j = i + 1; j < nzones && zones[j].z_name == NULL; ++j)
			continue;
		outzone(&zones[i], j - i);
	}
	/*
	** Make links.
	*/
	for (i = 0; i < nlinks; ++i) {
		eat(links[i].l_filename, links[i].l_linenum);
		dolink(links[i].l_from, links[i].l_to, false);
		if (noise)
			for (j = 0; j < nlinks; ++j)
				if (strcmp(links[i].l_to,
					links[j].l_from) == 0)
						warning(_("link to link"));
	}
	if (lcltime != NULL) {
		eat(_("command line"), 1);
		dolink(lcltime, tzdefault, true);
	}
	if (psxrules != NULL) {
		eat(_("command line"), 1);
		dolink(psxrules, TZDEFRULES, true);
	}
	if (warnings && (ferror(stderr) || fclose(stderr) != 0))
	  return EXIT_FAILURE;
	return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

static bool
componentcheck(char const *name, char const *component,
	       char const *component_end)
{
	enum { component_len_max = 14 };
	ptrdiff_t component_len = component_end - component;
	if (component_len == 0) {
	  if (!*name)
	    error (_("empty file name"));
	  else
	    error (_(component == name
		     ? "file name '%s' begins with '/'"
		     : *component_end
		     ? "file name '%s' contains '//'"
		     : "file name '%s' ends with '/'"),
		   name);
	  return false;
	}
	if (0 < component_len && component_len <= 2
	    && component[0] == '.' && component_end[-1] == '.') {
	  int len = component_len;
	  error(_("file name '%s' contains '%.*s' component"),
		name, len, component);
	  return false;
	}
	if (noise) {
	  if (0 < component_len && component[0] == '-')
	    warning(_("file name '%s' component contains leading '-'"),
		    name);
	  if (component_len_max < component_len)
	    warning(_("file name '%s' contains overlength component"
		      " '%.*s...'"),
		    name, component_len_max, component);
	}
	return true;
}

static bool
namecheck(const char *name)
{
	register char const *cp;

	/* Benign characters in a portable file name.  */
	static char const benign[] =
	  "-/_"
	  "abcdefghijklmnopqrstuvwxyz"
	  "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

	/* Non-control chars in the POSIX portable character set,
	   excluding the benign characters.  */
	static char const printable_and_not_benign[] =
	  " !\"#$%&'()*+,.0123456789:;<=>?@[\\]^`{|}~";

	register char const *component = name;
	for (cp = name; *cp; cp++) {
		unsigned char c = *cp;
		if (noise && !strchr(benign, c)) {
			warning((strchr(printable_and_not_benign, c)
				 ? _("file name '%s' contains byte '%c'")
				 : _("file name '%s' contains byte '\\%o'")),
				name, c);
		}
		if (c == '/') {
			if (!componentcheck(name, component, cp))
			  return false;
			component = cp + 1;
		}
	}
	return componentcheck(name, component, cp);
}

/* Create symlink contents suitable for symlinking FROM to TO, as a
   freshly allocated string.  FROM should be a relative file name, and
   is relative to the global variable DIRECTORY.  TO can be either
   relative or absolute.  */
static char *
relname(char const *from, char const *to)
{
  size_t i, taillen, dotdotetcsize;
  size_t dir_len = 0, dotdots = 0, linksize = SIZE_MAX;
  char const *f = from;
  char *result = NULL;
  if (*to == '/') {
    /* Make F absolute too.  */
    size_t len = strlen(directory);
    bool needslash = len && directory[len - 1] != '/';
    linksize = len + needslash + strlen(from) + 1;
    f = result = emalloc(linksize);
    strcpy(result, directory);
    result[len] = '/';
    strcpy(result + len + needslash, from);
  }
  for (i = 0; f[i] && f[i] == to[i]; i++)
    if (f[i] == '/')
      dir_len = i + 1;
  for (; to[i]; i++)
    dotdots += to[i] == '/' && to[i - 1] != '/';
  taillen = strlen(f + dir_len);
  dotdotetcsize = 3 * dotdots + taillen + 1;
  if (dotdotetcsize <= linksize) {
    if (!result)
      result = emalloc(dotdotetcsize);
    for (i = 0; i < dotdots; i++)
      memcpy(result + 3 * i, "../", 3);
    memmove(result + 3 * dotdots, f + dir_len, taillen + 1);
  }
  return result;
}

/* Hard link FROM to TO, following any symbolic links.
   Return 0 if successful, an error number otherwise.  */
static int
hardlinkerr(char const *from, char const *to)
{
  int r = linkat(AT_FDCWD, from, AT_FDCWD, to, AT_SYMLINK_FOLLOW);
  return r == 0 ? 0 : errno;
}

static void
dolink(char const *fromfield, char const *tofield, bool staysymlink)
{
	bool todirs_made = false;
	int link_errno;

	/*
	** We get to be careful here since
	** there's a fair chance of root running us.
	*/
	if (itsdir(fromfield)) {
		fprintf(stderr, _("%s: link from %s/%s failed: %s\n"),
			progname, directory, fromfield, strerror(EPERM));
		exit(EXIT_FAILURE);
	}
	if (staysymlink)
	  staysymlink = itssymlink(tofield);
	if (remove(tofield) == 0)
	  todirs_made = true;
	else if (errno != ENOENT) {
	  char const *e = strerror(errno);
	  fprintf(stderr, _("%s: Can't remove %s/%s: %s\n"),
		  progname, directory, tofield, e);
	  exit(EXIT_FAILURE);
	}
	link_errno = staysymlink ? ENOTSUP : hardlinkerr(fromfield, tofield);
	if (link_errno == ENOENT && !todirs_made) {
	  mkdirs(tofield, true);
	  todirs_made = true;
	  link_errno = hardlinkerr(fromfield, tofield);
	}
	if (link_errno != 0) {
	  bool absolute = *fromfield == '/';
	  char *linkalloc = absolute ? NULL : relname(fromfield, tofield);
	  char const *contents = absolute ? fromfield : linkalloc;
	  int symlink_errno = symlink(contents, tofield) == 0 ? 0 : errno;
	  if (!todirs_made
	      && (symlink_errno == ENOENT || symlink_errno == ENOTSUP)) {
	    mkdirs(tofield, true);
	    if (symlink_errno == ENOENT)
	      symlink_errno = symlink(contents, tofield) == 0 ? 0 : errno;
	  }
	  free(linkalloc);
	  if (symlink_errno == 0) {
	    if (link_errno != ENOTSUP)
	      warning(_("symbolic link used because hard link failed: %s"),
		      strerror(link_errno));
	  } else {
	    FILE *fp, *tp;
	    int c;
	    fp = fopen(fromfield, "rb");
	    if (!fp) {
	      char const *e = strerror(errno);
	      fprintf(stderr, _("%s: Can't read %s/%s: %s\n"),
		      progname, directory, fromfield, e);
	      exit(EXIT_FAILURE);
	    }
	    tp = fopen(tofield, "wb");
	    if (!tp) {
	      char const *e = strerror(errno);
	      fprintf(stderr, _("%s: Can't create %s/%s: %s\n"),
		      progname, directory, tofield, e);
	      exit(EXIT_FAILURE);
	    }
	    while ((c = getc(fp)) != EOF)
	      putc(c, tp);
	    close_file(fp, directory, fromfield);
	    close_file(tp, directory, tofield);
	    if (link_errno != ENOTSUP)
	      warning(_("copy used because hard link failed: %s"),
		      strerror(link_errno));
	    else if (symlink_errno != ENOTSUP)
	      warning(_("copy used because symbolic link failed: %s"),
		      strerror(symlink_errno));
	  }
	}
}

/* Return true if NAME is a directory.  */
static bool
itsdir(char const *name)
{
	struct stat st;
	int res = stat(name, &st);
#ifdef S_ISDIR
	if (res == 0)
		return S_ISDIR(st.st_mode) != 0;
#endif
	if (res == 0 || errno == EOVERFLOW) {
		size_t n = strlen(name);
		char *nameslashdot = emalloc(n + 3);
		bool dir;
		memcpy(nameslashdot, name, n);
		strcpy(&nameslashdot[n], &"/."[! (n && name[n - 1] != '/')]);
		dir = stat(nameslashdot, &st) == 0 || errno == EOVERFLOW;
		free(nameslashdot);
		return dir;
	}
	return false;
}

/* Return true if NAME is a symbolic link.  */
static bool
itssymlink(char const *name)
{
  char c;
  return 0 <= readlink(name, &c, 1);
}

/*
** Associate sets of rules with zones.
*/

/*
** Sort by rule name.
*/

static int
rcomp(const void *cp1, const void *cp2)
{
	return strcmp(((const struct rule *) cp1)->r_name,
		((const struct rule *) cp2)->r_name);
}

static void
associate(void)
{
	register struct zone *	zp;
	register struct rule *	rp;
	register ptrdiff_t i, j, base, out;

	if (nrules != 0) {
		qsort(rules, nrules, sizeof *rules, rcomp);
		for (i = 0; i < nrules - 1; ++i) {
			if (strcmp(rules[i].r_name,
				rules[i + 1].r_name) != 0)
					continue;
			if (strcmp(rules[i].r_filename,
				rules[i + 1].r_filename) == 0)
					continue;
			eat(rules[i].r_filename, rules[i].r_linenum);
			warning(_("same rule name in multiple files"));
			eat(rules[i + 1].r_filename, rules[i + 1].r_linenum);
			warning(_("same rule name in multiple files"));
			for (j = i + 2; j < nrules; ++j) {
				if (strcmp(rules[i].r_name,
					rules[j].r_name) != 0)
						break;
				if (strcmp(rules[i].r_filename,
					rules[j].r_filename) == 0)
						continue;
				if (strcmp(rules[i + 1].r_filename,
					rules[j].r_filename) == 0)
						continue;
				break;
			}
			i = j - 1;
		}
	}
	for (i = 0; i < nzones; ++i) {
		zp = &zones[i];
		zp->z_rules = NULL;
		zp->z_nrules = 0;
	}
	for (base = 0; base < nrules; base = out) {
		rp = &rules[base];
		for (out = base + 1; out < nrules; ++out)
			if (strcmp(rp->r_name, rules[out].r_name) != 0)
				break;
		for (i = 0; i < nzones; ++i) {
			zp = &zones[i];
			if (strcmp(zp->z_rule, rp->r_name) != 0)
				continue;
			zp->z_rules = rp;
			zp->z_nrules = out - base;
		}
	}
	for (i = 0; i < nzones; ++i) {
		zp = &zones[i];
		if (zp->z_nrules == 0) {
			/*
			** Maybe we have a local standard time offset.
			*/
			eat(zp->z_filename, zp->z_linenum);
			zp->z_save = getsave(zp->z_rule, &zp->z_isdst);
			/*
			** Note, though, that if there's no rule,
			** a '%s' in the format is a bad thing.
			*/
			if (zp->z_format_specifier == 's')
				error("%s", _("%s in ruleless zone"));
		}
	}
	if (errors)
		exit(EXIT_FAILURE);
}

static void
infile(const char *name)
{
	register FILE *			fp;
	register char **		fields;
	register char *			cp;
	register const struct lookup *	lp;
	register int			nfields;
	register bool			wantcont;
	register lineno			num;
	char				buf[BUFSIZ];

	if (strcmp(name, "-") == 0) {
		name = _("standard input");
		fp = stdin;
	} else if ((fp = fopen(name, "r")) == NULL) {
		const char *e = strerror(errno);

		fprintf(stderr, _("%s: Can't open %s: %s\n"),
			progname, name, e);
		exit(EXIT_FAILURE);
	}
	wantcont = false;
	for (num = 1; ; ++num) {
		eat(name, num);
		if (fgets(buf, sizeof buf, fp) != buf)
			break;
		cp = strchr(buf, '\n');
		if (cp == NULL) {
			error(_("line too long"));
			exit(EXIT_FAILURE);
		}
		*cp = '\0';
		fields = getfields(buf);
		nfields = 0;
		while (fields[nfields] != NULL) {
			static char	nada;

			if (strcmp(fields[nfields], "-") == 0)
				fields[nfields] = &nada;
			++nfields;
		}
		if (nfields == 0) {
		  if (name == leapsec && *buf == '#')
		    sscanf(buf, "#expires %"SCNdZIC, &comment_leapexpires);
		} else if (wantcont) {
			wantcont = inzcont(fields, nfields);
		} else {
			struct lookup const *line_codes
			  = name == leapsec ? leap_line_codes : zi_line_codes;
			lp = byword(fields[0], line_codes);
			if (lp == NULL)
				error(_("input line of unknown type"));
			else switch (lp->l_value) {
				case LC_RULE:
					inrule(fields, nfields);
					wantcont = false;
					break;
				case LC_ZONE:
					wantcont = inzone(fields, nfields);
					break;
				case LC_LINK:
					inlink(fields, nfields);
					wantcont = false;
					break;
				case LC_LEAP:
					inleap(fields, nfields);
					wantcont = false;
					break;
				case LC_EXPIRES:
					inexpires(fields, nfields);
					wantcont = false;
					break;
				default:	/* "cannot happen" */
					fprintf(stderr,
_("%s: panic: Invalid l_value %d\n"),
						progname, lp->l_value);
					exit(EXIT_FAILURE);
			}
		}
		free(fields);
	}
	close_file(fp, NULL, filename);
	if (wantcont)
		error(_("expected continuation line not found"));
}

/*
** Convert a string of one of the forms
**	h	-h	hh:mm	-hh:mm	hh:mm:ss	-hh:mm:ss
** into a number of seconds.
** A null string maps to zero.
** Call error with errstring and return zero on errors.
*/

static zic_t
gethms(char const *string, char const *errstring)
{
	zic_t	hh;
	int sign, mm = 0, ss = 0;
	char hhx, mmx, ssx, xr = '0', xs;
	int tenths = 0;
	bool ok = true;

	if (string == NULL || *string == '\0')
		return 0;
	if (*string == '-') {
		sign = -1;
		++string;
	} else	sign = 1;
	switch (sscanf(string,
		       "%"SCNdZIC"%c%d%c%d%c%1d%*[0]%c%*[0123456789]%c",
		       &hh, &hhx, &mm, &mmx, &ss, &ssx, &tenths, &xr, &xs)) {
	  default: ok = false; break;
	  case 8:
	    ok = '0' <= xr && xr <= '9';
	    /* fallthrough */
	  case 7:
	    ok &= ssx == '.';
	    if (ok && noise)
	      warning(_("fractional seconds rejected by"
			" pre-2018 versions of zic"));
	    /* fallthrough */
	  case 5: ok &= mmx == ':'; /* fallthrough */
	  case 3: ok &= hhx == ':'; /* fallthrough */
	  case 1: break;
	}
	if (!ok) {
			error("%s", errstring);
			return 0;
	}
	if (hh < 0 ||
		mm < 0 || mm >= MINSPERHOUR ||
		ss < 0 || ss > SECSPERMIN) {
			error("%s", errstring);
			return 0;
	}
	if (ZIC_MAX / SECSPERHOUR < hh) {
		error(_("time overflow"));
		return 0;
	}
	ss += 5 + ((ss ^ 1) & (xr == '0')) <= tenths; /* Round to even.  */
	if (noise && (hh > HOURSPERDAY ||
		(hh == HOURSPERDAY && (mm != 0 || ss != 0))))
warning(_("values over 24 hours not handled by pre-2007 versions of zic"));
	return oadd(sign * hh * SECSPERHOUR,
		    sign * (mm * SECSPERMIN + ss));
}

static zic_t
getsave(char *field, bool *isdst)
{
  int dst = -1;
  zic_t save;
  size_t fieldlen = strlen(field);
  if (fieldlen != 0) {
    char *ep = field + fieldlen - 1;
    switch (*ep) {
      case 'd': dst = 1; *ep = '\0'; break;
      case 's': dst = 0; *ep = '\0'; break;
    }
  }
  save = gethms(field, _("invalid saved time"));
  *isdst = dst < 0 ? save != 0 : dst;
  return save;
}

static void
inrule(char **fields, int nfields)
{
	static struct rule	r;

	if (nfields != RULE_FIELDS) {
		error(_("wrong number of fields on Rule line"));
		return;
	}
	switch (*fields[RF_NAME]) {
	  case '\0':
	  case ' ': case '\f': case '\n': case '\r': case '\t': case '\v':
	  case '+': case '-':
	  case '0': case '1': case '2': case '3': case '4':
	  case '5': case '6': case '7': case '8': case '9':
		error(_("Invalid rule name \"%s\""), fields[RF_NAME]);
		return;
	}
	r.r_filename = filename;
	r.r_linenum = linenum;
	r.r_save = getsave(fields[RF_SAVE], &r.r_isdst);
	rulesub(&r, fields[RF_LOYEAR], fields[RF_HIYEAR], fields[RF_COMMAND],
		fields[RF_MONTH], fields[RF_DAY], fields[RF_TOD]);
	r.r_name = ecpyalloc(fields[RF_NAME]);
	r.r_abbrvar = ecpyalloc(fields[RF_ABBRVAR]);
	if (max_abbrvar_len < strlen(r.r_abbrvar))
		max_abbrvar_len = strlen(r.r_abbrvar);
	rules = growalloc(rules, sizeof *rules, nrules, &nrules_alloc);
	rules[nrules++] = r;
}

static bool
inzone(char **fields, int nfields)
{
	register ptrdiff_t i;

	if (nfields < ZONE_MINFIELDS || nfields > ZONE_MAXFIELDS) {
		error(_("wrong number of fields on Zone line"));
		return false;
	}
	if (lcltime != NULL && strcmp(fields[ZF_NAME], tzdefault) == 0) {
		error(
_("\"Zone %s\" line and -l option are mutually exclusive"),
			tzdefault);
		return false;
	}
	if (strcmp(fields[ZF_NAME], TZDEFRULES) == 0 && psxrules != NULL) {
		error(
_("\"Zone %s\" line and -p option are mutually exclusive"),
			TZDEFRULES);
		return false;
	}
	for (i = 0; i < nzones; ++i)
		if (zones[i].z_name != NULL &&
			strcmp(zones[i].z_name, fields[ZF_NAME]) == 0) {
				error(_("duplicate zone name %s"
					" (file \"%s\", line %"PRIdMAX")"),
					fields[ZF_NAME],
					zones[i].z_filename,
					zones[i].z_linenum);
				return false;
		}
	return inzsub(fields, nfields, false);
}

static bool
inzcont(char **fields, int nfields)
{
	if (nfields < ZONEC_MINFIELDS || nfields > ZONEC_MAXFIELDS) {
		error(_("wrong number of fields on Zone continuation line"));
		return false;
	}
	return inzsub(fields, nfields, true);
}

static bool
inzsub(char **fields, int nfields, bool iscont)
{
	register char *		cp;
	char *			cp1;
	static struct zone	z;
	register int		i_stdoff, i_rule, i_format;
	register int		i_untilyear, i_untilmonth;
	register int		i_untilday, i_untiltime;
	register bool		hasuntil;

	if (iscont) {
		i_stdoff = ZFC_STDOFF;
		i_rule = ZFC_RULE;
		i_format = ZFC_FORMAT;
		i_untilyear = ZFC_TILYEAR;
		i_untilmonth = ZFC_TILMONTH;
		i_untilday = ZFC_TILDAY;
		i_untiltime = ZFC_TILTIME;
		z.z_name = NULL;
	} else if (!namecheck(fields[ZF_NAME]))
		return false;
	else {
		i_stdoff = ZF_STDOFF;
		i_rule = ZF_RULE;
		i_format = ZF_FORMAT;
		i_untilyear = ZF_TILYEAR;
		i_untilmonth = ZF_TILMONTH;
		i_untilday = ZF_TILDAY;
		i_untiltime = ZF_TILTIME;
		z.z_name = ecpyalloc(fields[ZF_NAME]);
	}
	z.z_filename = filename;
	z.z_linenum = linenum;
	z.z_stdoff = gethms(fields[i_stdoff], _("invalid UT offset"));
	if ((cp = strchr(fields[i_format], '%')) != 0) {
		if ((*++cp != 's' && *cp != 'z') || strchr(cp, '%')
		    || strchr(fields[i_format], '/')) {
			error(_("invalid abbreviation format"));
			return false;
		}
	}
	z.z_rule = ecpyalloc(fields[i_rule]);
	z.z_format = cp1 = ecpyalloc(fields[i_format]);
	z.z_format_specifier = cp ? *cp : '\0';
	if (z.z_format_specifier == 'z') {
	  if (noise)
	    warning(_("format '%s' not handled by pre-2015 versions of zic"),
		    z.z_format);
	  cp1[cp - fields[i_format]] = 's';
	}
	if (max_format_len < strlen(z.z_format))
		max_format_len = strlen(z.z_format);
	hasuntil = nfields > i_untilyear;
	if (hasuntil) {
		z.z_untilrule.r_filename = filename;
		z.z_untilrule.r_linenum = linenum;
		rulesub(&z.z_untilrule,
			fields[i_untilyear],
			"only",
			"",
			(nfields > i_untilmonth) ?
			fields[i_untilmonth] : "Jan",
			(nfields > i_untilday) ? fields[i_untilday] : "1",
			(nfields > i_untiltime) ? fields[i_untiltime] : "0");
		z.z_untiltime = rpytime(&z.z_untilrule,
			z.z_untilrule.r_loyear);
		if (iscont && nzones > 0 &&
			z.z_untiltime > min_time &&
			z.z_untiltime < max_time &&
			zones[nzones - 1].z_untiltime > min_time &&
			zones[nzones - 1].z_untiltime < max_time &&
			zones[nzones - 1].z_untiltime >= z.z_untiltime) {
				error(_(
"Zone continuation line end time is not after end time of previous line"
					));
				return false;
		}
	}
	zones = growalloc(zones, sizeof *zones, nzones, &nzones_alloc);
	zones[nzones++] = z;
	/*
	** If there was an UNTIL field on this line,
	** there's more information about the zone on the next line.
	*/
	return hasuntil;
}

static zic_t
getleapdatetime(char **fields, int nfields, bool expire_line)
{
	register const char *		cp;
	register const struct lookup *	lp;
	register zic_t			i, j;
	zic_t				year;
	int				month, day;
	zic_t				dayoff, tod;
	zic_t				t;
	char xs;

	dayoff = 0;
	cp = fields[LP_YEAR];
	if (sscanf(cp, "%"SCNdZIC"%c", &year, &xs) != 1) {
		/*
		** Leapin' Lizards!
		*/
		error(_("invalid leaping year"));
		return -1;
	}
	if (!expire_line) {
	    if (!leapseen || leapmaxyear < year)
		leapmaxyear = year;
	    if (!leapseen || leapminyear > year)
		leapminyear = year;
	    leapseen = true;
	}
	j = EPOCH_YEAR;
	while (j != year) {
		if (year > j) {
			i = len_years[isleap(j)];
			++j;
		} else {
			--j;
			i = -len_years[isleap(j)];
		}
		dayoff = oadd(dayoff, i);
	}
	if ((lp = byword(fields[LP_MONTH], mon_names)) == NULL) {
		error(_("invalid month name"));
		return -1;
	}
	month = lp->l_value;
	j = TM_JANUARY;
	while (j != month) {
		i = len_months[isleap(year)][j];
		dayoff = oadd(dayoff, i);
		++j;
	}
	cp = fields[LP_DAY];
	if (sscanf(cp, "%d%c", &day, &xs) != 1 ||
		day <= 0 || day > len_months[isleap(year)][month]) {
			error(_("invalid day of month"));
			return -1;
	}
	dayoff = oadd(dayoff, day - 1);
	if (dayoff < min_time / SECSPERDAY) {
		error(_("time too small"));
		return -1;
	}
	if (dayoff > max_time / SECSPERDAY) {
		error(_("time too large"));
		return -1;
	}
	t = dayoff * SECSPERDAY;
	tod = gethms(fields[LP_TIME], _("invalid time of day"));
	t = tadd(t, tod);
	if (t < 0)
	  error(_("leap second precedes Epoch"));
	return t;
}

static void
inleap(char **fields, int nfields)
{
  if (nfields != LEAP_FIELDS)
    error(_("wrong number of fields on Leap line"));
  else {
    zic_t t = getleapdatetime(fields, nfields, false);
    if (0 <= t) {
      struct lookup const *lp = byword(fields[LP_ROLL], leap_types);
      if (!lp)
	error(_("invalid Rolling/Stationary field on Leap line"));
      else {
	int correction = 0;
	if (!fields[LP_CORR][0]) /* infile() turns "-" into "".  */
	  correction = -1;
	else if (strcmp(fields[LP_CORR], "+") == 0)
	  correction = 1;
	else
	  error(_("invalid CORRECTION field on Leap line"));
	if (correction)
	  leapadd(t, correction, lp->l_value);
      }
    }
  }
}

static void
inexpires(char **fields, int nfields)
{
  if (nfields != EXPIRES_FIELDS)
    error(_("wrong number of fields on Expires line"));
  else if (0 <= leapexpires)
    error(_("multiple Expires lines"));
  else
    leapexpires = getleapdatetime(fields, nfields, true);
}

static void
inlink(char **fields, int nfields)
{
	struct link	l;

	if (nfields != LINK_FIELDS) {
		error(_("wrong number of fields on Link line"));
		return;
	}
	if (*fields[LF_FROM] == '\0') {
		error(_("blank FROM field on Link line"));
		return;
	}
	if (! namecheck(fields[LF_TO]))
	  return;
	l.l_filename = filename;
	l.l_linenum = linenum;
	l.l_from = ecpyalloc(fields[LF_FROM]);
	l.l_to = ecpyalloc(fields[LF_TO]);
	links = growalloc(links, sizeof *links, nlinks, &nlinks_alloc);
	links[nlinks++] = l;
}

static void
rulesub(struct rule *rp, const char *loyearp, const char *hiyearp,
	const char *typep, const char *monthp, const char *dayp,
	const char *timep)
{
	register const struct lookup *	lp;
	register const char *		cp;
	register char *			dp;
	register char *			ep;
	char xs;

	if ((lp = byword(monthp, mon_names)) == NULL) {
		error(_("invalid month name"));
		return;
	}
	rp->r_month = lp->l_value;
	rp->r_todisstd = false;
	rp->r_todisut = false;
	dp = ecpyalloc(timep);
	if (*dp != '\0') {
		ep = dp + strlen(dp) - 1;
		switch (lowerit(*ep)) {
			case 's':	/* Standard */
				rp->r_todisstd = true;
				rp->r_todisut = false;
				*ep = '\0';
				break;
			case 'w':	/* Wall */
				rp->r_todisstd = false;
				rp->r_todisut = false;
				*ep = '\0';
				break;
			case 'g':	/* Greenwich */
			case 'u':	/* Universal */
			case 'z':	/* Zulu */
				rp->r_todisstd = true;
				rp->r_todisut = true;
				*ep = '\0';
				break;
		}
	}
	rp->r_tod = gethms(dp, _("invalid time of day"));
	free(dp);
	/*
	** Year work.
	*/
	cp = loyearp;
	lp = byword(cp, begin_years);
	rp->r_lowasnum = lp == NULL;
	if (!rp->r_lowasnum) switch (lp->l_value) {
		case YR_MINIMUM:
			rp->r_loyear = ZIC_MIN;
			break;
		case YR_MAXIMUM:
			rp->r_loyear = ZIC_MAX;
			break;
		default:	/* "cannot happen" */
			fprintf(stderr,
				_("%s: panic: Invalid l_value %d\n"),
				progname, lp->l_value);
			exit(EXIT_FAILURE);
	} else if (sscanf(cp, "%"SCNdZIC"%c", &rp->r_loyear, &xs) != 1) {
		error(_("invalid starting year"));
		return;
	}
	cp = hiyearp;
	lp = byword(cp, end_years);
	rp->r_hiwasnum = lp == NULL;
	if (!rp->r_hiwasnum) switch (lp->l_value) {
		case YR_MINIMUM:
			rp->r_hiyear = ZIC_MIN;
			break;
		case YR_MAXIMUM:
			rp->r_hiyear = ZIC_MAX;
			break;
		case YR_ONLY:
			rp->r_hiyear = rp->r_loyear;
			break;
		default:	/* "cannot happen" */
			fprintf(stderr,
				_("%s: panic: Invalid l_value %d\n"),
				progname, lp->l_value);
			exit(EXIT_FAILURE);
	} else if (sscanf(cp, "%"SCNdZIC"%c", &rp->r_hiyear, &xs) != 1) {
		error(_("invalid ending year"));
		return;
	}
	if (rp->r_loyear > rp->r_hiyear) {
		error(_("starting year greater than ending year"));
		return;
	}
	if (*typep == '\0')
		rp->r_yrtype = NULL;
	else {
		if (rp->r_loyear == rp->r_hiyear) {
			error(_("typed single year"));
			return;
		}
		warning(_("year type \"%s\" is obsolete; use \"-\" instead"),
			typep);
		rp->r_yrtype = ecpyalloc(typep);
	}
	/*
	** Day work.
	** Accept things such as:
	**	1
	**	lastSunday
	**	last-Sunday (undocumented; warn about this)
	**	Sun<=20
	**	Sun>=7
	*/
	dp = ecpyalloc(dayp);
	if ((lp = byword(dp, lasts)) != NULL) {
		rp->r_dycode = DC_DOWLEQ;
		rp->r_wday = lp->l_value;
		rp->r_dayofmonth = len_months[1][rp->r_month];
	} else {
		if ((ep = strchr(dp, '<')) != 0)
			rp->r_dycode = DC_DOWLEQ;
		else if ((ep = strchr(dp, '>')) != 0)
			rp->r_dycode = DC_DOWGEQ;
		else {
			ep = dp;
			rp->r_dycode = DC_DOM;
		}
		if (rp->r_dycode != DC_DOM) {
			*ep++ = 0;
			if (*ep++ != '=') {
				error(_("invalid day of month"));
				free(dp);
				return;
			}
			if ((lp = byword(dp, wday_names)) == NULL) {
				error(_("invalid weekday name"));
				free(dp);
				return;
			}
			rp->r_wday = lp->l_value;
		}
		if (sscanf(ep, "%d%c", &rp->r_dayofmonth, &xs) != 1 ||
			rp->r_dayofmonth <= 0 ||
			(rp->r_dayofmonth > len_months[1][rp->r_month])) {
				error(_("invalid day of month"));
				free(dp);
				return;
		}
	}
	free(dp);
}

static void
convert(const int_fast32_t val, char *const buf)
{
	register int	i;
	register int	shift;
	unsigned char *const b = (unsigned char *) buf;

	for (i = 0, shift = 24; i < 4; ++i, shift -= 8)
		b[i] = val >> shift;
}

static void
convert64(const zic_t val, char *const buf)
{
	register int	i;
	register int	shift;
	unsigned char *const b = (unsigned char *) buf;

	for (i = 0, shift = 56; i < 8; ++i, shift -= 8)
		b[i] = val >> shift;
}

static void
puttzcode(const int_fast32_t val, FILE *const fp)
{
	char	buf[4];

	convert(val, buf);
	fwrite(buf, sizeof buf, 1, fp);
}

static void
puttzcodepass(zic_t val, FILE *fp, int pass)
{
  if (pass == 1)
    puttzcode(val, fp);
  else {
	char	buf[8];

	convert64(val, buf);
	fwrite(buf, sizeof buf, 1, fp);
  }
}

static int
atcomp(const void *avp, const void *bvp)
{
	const zic_t	a = ((const struct attype *) avp)->at;
	const zic_t	b = ((const struct attype *) bvp)->at;

	return (a < b) ? -1 : (a > b);
}

struct timerange {
  int defaulttype;
  ptrdiff_t base, count;
  int leapbase, leapcount;
};

static struct timerange
limitrange(struct timerange r, zic_t lo, zic_t hi,
	   zic_t const *ats, unsigned char const *types)
{
  while (0 < r.count && ats[r.base] < lo) {
    r.defaulttype = types[r.base];
    r.count--;
    r.base++;
  }
  while (0 < r.leapcount && trans[r.leapbase] < lo) {
    r.leapcount--;
    r.leapbase++;
  }

  if (hi < ZIC_MAX) {
    while (0 < r.count && hi + 1 < ats[r.base + r.count - 1])
      r.count--;
    while (0 < r.leapcount && hi + 1 < trans[r.leapbase + r.leapcount - 1])
      r.leapcount--;
  }

  return r;
}

static void
writezone(const char *const name, const char *const string, char version,
	  int defaulttype)
{
	register FILE *			fp;
	register ptrdiff_t		i, j;
	register int			pass;
	static const struct tzhead	tzh0;
	static struct tzhead		tzh;
	bool dir_checked = false;
	zic_t one = 1;
	zic_t y2038_boundary = one << 31;
	ptrdiff_t nats = timecnt + WORK_AROUND_QTBUG_53071;

	/* Allocate the ATS and TYPES arrays via a single malloc,
	   as this is a bit faster.  */
	zic_t *ats = emalloc(align_to(size_product(nats, sizeof *ats + 1),
				      _Alignof(zic_t)));
	void *typesptr = ats + nats;
	unsigned char *types = typesptr;
	struct timerange rangeall, range32, range64;

	/*
	** Sort.
	*/
	if (timecnt > 1)
		qsort(attypes, timecnt, sizeof *attypes, atcomp);
	/*
	** Optimize.
	*/
	{
		ptrdiff_t fromi, toi;

		toi = 0;
		fromi = 0;
		for ( ; fromi < timecnt; ++fromi) {
			if (toi != 0
			    && ((attypes[fromi].at
				 + utoffs[attypes[toi - 1].type])
				<= (attypes[toi - 1].at
				    + utoffs[toi == 1 ? 0
					     : attypes[toi - 2].type]))) {
					attypes[toi - 1].type =
						attypes[fromi].type;
					continue;
			}
			if (toi == 0
			    || attypes[fromi].dontmerge
			    || (utoffs[attypes[toi - 1].type]
				!= utoffs[attypes[fromi].type])
			    || (isdsts[attypes[toi - 1].type]
				!= isdsts[attypes[fromi].type])
			    || (desigidx[attypes[toi - 1].type]
				!= desigidx[attypes[fromi].type]))
					attypes[toi++] = attypes[fromi];
		}
		timecnt = toi;
	}

	if (noise && timecnt > 1200) {
	  if (timecnt > TZ_MAX_TIMES)
		warning(_("reference clients mishandle"
			  " more than %d transition times"),
			TZ_MAX_TIMES);
	  else
		warning(_("pre-2014 clients may mishandle"
			  " more than 1200 transition times"));
	}
	/*
	** Transfer.
	*/
	for (i = 0; i < timecnt; ++i) {
		ats[i] = attypes[i].at;
		types[i] = attypes[i].type;
	}

	/*
	** Correct for leap seconds.
	*/
	for (i = 0; i < timecnt; ++i) {
		j = leapcnt;
		while (--j >= 0)
			if (ats[i] > trans[j] - corr[j]) {
				ats[i] = tadd(ats[i], corr[j]);
				break;
			}
	}

	/* Work around QTBUG-53071 for timestamps less than y2038_boundary - 1,
	   by inserting a no-op transition at time y2038_boundary - 1.
	   This works only for timestamps before the boundary, which
	   should be good enough in practice as QTBUG-53071 should be
	   long-dead by 2038.  Do this after correcting for leap
	   seconds, as the idea is to insert a transition just before
	   32-bit time_t rolls around, and this occurs at a slightly
	   different moment if transitions are leap-second corrected.  */
	if (WORK_AROUND_QTBUG_53071 && timecnt != 0 && want_bloat()
	    && ats[timecnt - 1] < y2038_boundary - 1 && strchr(string, '<')) {
	  ats[timecnt] = y2038_boundary - 1;
	  types[timecnt] = types[timecnt - 1];
	  timecnt++;
	}

	rangeall.defaulttype = defaulttype;
	rangeall.base = rangeall.leapbase = 0;
	rangeall.count = timecnt;
	rangeall.leapcount = leapcnt;
	range64 = limitrange(rangeall, lo_time, hi_time, ats, types);
	range32 = limitrange(range64, INT32_MIN, INT32_MAX, ats, types);

	/*
	** Remove old file, if any, to snap links.
	*/
	if (remove(name) == 0)
		dir_checked = true;
	else if (errno != ENOENT) {
		const char *e = strerror(errno);

		fprintf(stderr, _("%s: Can't remove %s/%s: %s\n"),
			progname, directory, name, e);
		exit(EXIT_FAILURE);
	}
	fp = fopen(name, "wb");
	if (!fp) {
	  int fopen_errno = errno;
	  if (fopen_errno == ENOENT && !dir_checked) {
	    mkdirs(name, true);
	    fp = fopen(name, "wb");
	    fopen_errno = errno;
	  }
	  if (!fp) {
	    fprintf(stderr, _("%s: Can't create %s/%s: %s\n"),
		    progname, directory, name, strerror(fopen_errno));
	    exit(EXIT_FAILURE);
	  }
	}
	for (pass = 1; pass <= 2; ++pass) {
		register ptrdiff_t thistimei, thistimecnt, thistimelim;
		register int	thisleapi, thisleapcnt, thisleaplim;
		int currenttype, thisdefaulttype;
		bool locut, hicut;
		zic_t lo;
		int old0;
		char		omittype[TZ_MAX_TYPES];
		int		typemap[TZ_MAX_TYPES];
		int		thistypecnt, stdcnt, utcnt;
		char		thischars[TZ_MAX_CHARS];
		int		thischarcnt;
		bool		toomanytimes;
		int		indmap[TZ_MAX_CHARS];

		if (pass == 1) {
			/* Arguably the default time type in the 32-bit data
			   should be range32.defaulttype, which is suited for
			   timestamps just before INT32_MIN.  However, zic
			   traditionally used the time type of the indefinite
			   past instead.  Internet RFC 8532 says readers should
			   ignore 32-bit data, so this discrepancy matters only
			   to obsolete readers where the traditional type might
			   be more appropriate even if it's "wrong".  So, use
			   the historical zic value, unless -r specifies a low
			   cutoff that excludes some 32-bit timestamps.  */
			thisdefaulttype = (lo_time <= INT32_MIN
					   ? range64.defaulttype
					   : range32.defaulttype);

			thistimei = range32.base;
			thistimecnt = range32.count;
			toomanytimes = thistimecnt >> 31 >> 1 != 0;
			thisleapi = range32.leapbase;
			thisleapcnt = range32.leapcount;
			locut = INT32_MIN < lo_time;
			hicut = hi_time < INT32_MAX;
		} else {
			thisdefaulttype = range64.defaulttype;
			thistimei = range64.base;
			thistimecnt = range64.count;
			toomanytimes = thistimecnt >> 31 >> 31 >> 2 != 0;
			thisleapi = range64.leapbase;
			thisleapcnt = range64.leapcount;
			locut = min_time < lo_time;
			hicut = hi_time < max_time;
		}
		if (toomanytimes)
		  error(_("too many transition times"));

		/* Keep the last too-low transition if no transition is
		   exactly at LO.  The kept transition will be output as
		   a LO "transition"; see "Output a LO_TIME transition"
		   below.  This is needed when the output is truncated at
		   the start, and is also useful when catering to buggy
		   32-bit clients that do not use time type 0 for
		   timestamps before the first transition.  */
		if (0 < thistimei && ats[thistimei] != lo_time) {
		  thistimei--;
		  thistimecnt++;
		  locut = false;
		}

		thistimelim = thistimei + thistimecnt;
		thisleaplim = thisleapi + thisleapcnt;
		if (thistimecnt != 0) {
		  if (ats[thistimei] == lo_time)
		    locut = false;
		  if (hi_time < ZIC_MAX && ats[thistimelim - 1] == hi_time + 1)
		    hicut = false;
		}
		memset(omittype, true, typecnt);
		omittype[thisdefaulttype] = false;
		for (i = thistimei; i < thistimelim; i++)
		  omittype[types[i]] = false;

		/* Reorder types to make THISDEFAULTTYPE type 0.
		   Use TYPEMAP to swap OLD0 and THISDEFAULTTYPE so that
		   THISDEFAULTTYPE appears as type 0 in the output instead
		   of OLD0.  TYPEMAP also omits unused types.  */
		old0 = strlen(omittype);

#ifndef LEAVE_SOME_PRE_2011_SYSTEMS_IN_THE_LURCH
		/*
		** For some pre-2011 systems: if the last-to-be-written
		** standard (or daylight) type has an offset different from the
		** most recently used offset,
		** append an (unused) copy of the most recently used type
		** (to help get global "altzone" and "timezone" variables
		** set correctly).
		*/
		if (want_bloat()) {
			register int	mrudst, mrustd, hidst, histd, type;

			hidst = histd = mrudst = mrustd = -1;
			for (i = thistimei; i < thistimelim; ++i)
				if (isdsts[types[i]])
					mrudst = types[i];
				else	mrustd = types[i];
			for (i = old0; i < typecnt; i++) {
			  int h = (i == old0 ? thisdefaulttype
				   : i == thisdefaulttype ? old0 : i);
			  if (!omittype[h]) {
			    if (isdsts[h])
			      hidst = i;
			    else
			      histd = i;
			  }
			}
			if (hidst >= 0 && mrudst >= 0 && hidst != mrudst &&
				utoffs[hidst] != utoffs[mrudst]) {
					isdsts[mrudst] = -1;
					type = addtype(utoffs[mrudst],
						&chars[desigidx[mrudst]],
						true,
						ttisstds[mrudst],
						ttisuts[mrudst]);
					isdsts[mrudst] = 1;
					omittype[type] = false;
			}
			if (histd >= 0 && mrustd >= 0 && histd != mrustd &&
				utoffs[histd] != utoffs[mrustd]) {
					isdsts[mrustd] = -1;
					type = addtype(utoffs[mrustd],
						&chars[desigidx[mrustd]],
						false,
						ttisstds[mrustd],
						ttisuts[mrustd]);
					isdsts[mrustd] = 0;
					omittype[type] = false;
			}
		}
#endif /* !defined LEAVE_SOME_PRE_2011_SYSTEMS_IN_THE_LURCH */
		thistypecnt = 0;
		for (i = old0; i < typecnt; i++)
		  if (!omittype[i])
		    typemap[i == old0 ? thisdefaulttype
			    : i == thisdefaulttype ? old0 : i]
		      = thistypecnt++;

		for (i = 0; i < sizeof indmap / sizeof indmap[0]; ++i)
			indmap[i] = -1;
		thischarcnt = stdcnt = utcnt = 0;
		for (i = old0; i < typecnt; i++) {
			register char *	thisabbr;

			if (omittype[i])
				continue;
			if (ttisstds[i])
			  stdcnt = thistypecnt;
			if (ttisuts[i])
			  utcnt = thistypecnt;
			if (indmap[desigidx[i]] >= 0)
				continue;
			thisabbr = &chars[desigidx[i]];
			for (j = 0; j < thischarcnt; ++j)
				if (strcmp(&thischars[j], thisabbr) == 0)
					break;
			if (j == thischarcnt) {
				strcpy(&thischars[thischarcnt], thisabbr);
				thischarcnt += strlen(thisabbr) + 1;
			}
			indmap[desigidx[i]] = j;
		}
		if (pass == 1 && !want_bloat()) {
		  utcnt = stdcnt = thisleapcnt = 0;
		  thistimecnt = - (locut + hicut);
		  thistypecnt = thischarcnt = 1;
		  thistimelim = thistimei;
		}
#define DO(field)	fwrite(tzh.field, sizeof tzh.field, 1, fp)
		tzh = tzh0;
		memcpy(tzh.tzh_magic, TZ_MAGIC, sizeof tzh.tzh_magic);
		tzh.tzh_version[0] = version;
		convert(utcnt, tzh.tzh_ttisutcnt);
		convert(stdcnt, tzh.tzh_ttisstdcnt);
		convert(thisleapcnt, tzh.tzh_leapcnt);
		convert(locut + thistimecnt + hicut, tzh.tzh_timecnt);
		convert(thistypecnt, tzh.tzh_typecnt);
		convert(thischarcnt, tzh.tzh_charcnt);
		DO(tzh_magic);
		DO(tzh_version);
		DO(tzh_reserved);
		DO(tzh_ttisutcnt);
		DO(tzh_ttisstdcnt);
		DO(tzh_leapcnt);
		DO(tzh_timecnt);
		DO(tzh_typecnt);
		DO(tzh_charcnt);
#undef DO
		if (pass == 1 && !want_bloat()) {
		  /* Output a minimal data block with just one time type.  */
		  puttzcode(0, fp);	/* utoff */
		  putc(0, fp);		/* dst */
		  putc(0, fp);		/* index of abbreviation */
		  putc(0, fp);		/* empty-string abbreviation */
		  continue;
		}

		/* Output a LO_TIME transition if needed; see limitrange.
		   But do not go below the minimum representable value
		   for this pass.  */
		lo = pass == 1 && lo_time < INT32_MIN ? INT32_MIN : lo_time;

		if (locut)
		  puttzcodepass(lo, fp, pass);
		for (i = thistimei; i < thistimelim; ++i) {
		  zic_t at = ats[i] < lo ? lo : ats[i];
		  puttzcodepass(at, fp, pass);
		}
		if (hicut)
		  puttzcodepass(hi_time + 1, fp, pass);
		currenttype = 0;
		if (locut)
		  putc(currenttype, fp);
		for (i = thistimei; i < thistimelim; ++i) {
		  currenttype = typemap[types[i]];
		  putc(currenttype, fp);
		}
		if (hicut)
		  putc(currenttype, fp);

		for (i = old0; i < typecnt; i++) {
		  int h = (i == old0 ? thisdefaulttype
			   : i == thisdefaulttype ? old0 : i);
		  if (!omittype[h]) {
		    puttzcode(utoffs[h], fp);
		    putc(isdsts[h], fp);
		    putc(indmap[desigidx[h]], fp);
		  }
		}
		if (thischarcnt != 0)
			fwrite(thischars, sizeof thischars[0],
				      thischarcnt, fp);
		for (i = thisleapi; i < thisleaplim; ++i) {
			register zic_t	todo;

			if (roll[i]) {
				if (timecnt == 0 || trans[i] < ats[0]) {
					j = 0;
					while (isdsts[j])
						if (++j >= typecnt) {
							j = 0;
							break;
						}
				} else {
					j = 1;
					while (j < timecnt &&
						trans[i] >= ats[j])
							++j;
					j = types[j - 1];
				}
				todo = tadd(trans[i], -utoffs[j]);
			} else	todo = trans[i];
			puttzcodepass(todo, fp, pass);
			puttzcode(corr[i], fp);
		}
		if (stdcnt != 0)
		  for (i = old0; i < typecnt; i++)
			if (!omittype[i])
				putc(ttisstds[i], fp);
		if (utcnt != 0)
		  for (i = old0; i < typecnt; i++)
			if (!omittype[i])
				putc(ttisuts[i], fp);
	}
	fprintf(fp, "\n%s\n", string);
	close_file(fp, directory, name);
	free(ats);
}

static char const *
abbroffset(char *buf, zic_t offset)
{
  char sign = '+';
  int seconds, minutes;

  if (offset < 0) {
    offset = -offset;
    sign = '-';
  }

  seconds = offset % SECSPERMIN;
  offset /= SECSPERMIN;
  minutes = offset % MINSPERHOUR;
  offset /= MINSPERHOUR;
  if (100 <= offset) {
    error(_("%%z UT offset magnitude exceeds 99:59:59"));
    return "%z";
  } else {
    char *p = buf;
    *p++ = sign;
    *p++ = '0' + offset / 10;
    *p++ = '0' + offset % 10;
    if (minutes | seconds) {
      *p++ = '0' + minutes / 10;
      *p++ = '0' + minutes % 10;
      if (seconds) {
	*p++ = '0' + seconds / 10;
	*p++ = '0' + seconds % 10;
      }
    }
    *p = '\0';
    return buf;
  }
}

static size_t
doabbr(char *abbr, struct zone const *zp, char const *letters,
       bool isdst, zic_t save, bool doquotes)
{
	register char *	cp;
	register char *	slashp;
	register size_t	len;
	char const *format = zp->z_format;

	slashp = strchr(format, '/');
	if (slashp == NULL) {
	  char letterbuf[PERCENT_Z_LEN_BOUND + 1];
	  if (zp->z_format_specifier == 'z')
	    letters = abbroffset(letterbuf, zp->z_stdoff + save);
	  else if (!letters)
	    letters = "%s";
	  sprintf(abbr, format, letters);
	} else if (isdst) {
		strcpy(abbr, slashp + 1);
	} else {
		memcpy(abbr, format, slashp - format);
		abbr[slashp - format] = '\0';
	}
	len = strlen(abbr);
	if (!doquotes)
		return len;
	for (cp = abbr; is_alpha(*cp); cp++)
		continue;
	if (len > 0 && *cp == '\0')
		return len;
	abbr[len + 2] = '\0';
	abbr[len + 1] = '>';
	memmove(abbr + 1, abbr, len);
	abbr[0] = '<';
	return len + 2;
}

static void
updateminmax(const zic_t x)
{
	if (min_year > x)
		min_year = x;
	if (max_year < x)
		max_year = x;
}

static int
stringoffset(char *result, zic_t offset)
{
	register int	hours;
	register int	minutes;
	register int	seconds;
	bool negative = offset < 0;
	int len = negative;

	if (negative) {
		offset = -offset;
		result[0] = '-';
	}
	seconds = offset % SECSPERMIN;
	offset /= SECSPERMIN;
	minutes = offset % MINSPERHOUR;
	offset /= MINSPERHOUR;
	hours = offset;
	if (hours >= HOURSPERDAY * DAYSPERWEEK) {
		result[0] = '\0';
		return 0;
	}
	len += sprintf(result + len, "%d", hours);
	if (minutes != 0 || seconds != 0) {
		len += sprintf(result + len, ":%02d", minutes);
		if (seconds != 0)
			len += sprintf(result + len, ":%02d", seconds);
	}
	return len;
}

static int
stringrule(char *result, struct rule *const rp, zic_t save, zic_t stdoff)
{
	register zic_t	tod = rp->r_tod;
	register int	compat = 0;

	if (rp->r_dycode == DC_DOM) {
		register int	month, total;

		if (rp->r_dayofmonth == 29 && rp->r_month == TM_FEBRUARY)
			return -1;
		total = 0;
		for (month = 0; month < rp->r_month; ++month)
			total += len_months[0][month];
		/* Omit the "J" in Jan and Feb, as that's shorter.  */
		if (rp->r_month <= 1)
		  result += sprintf(result, "%d", total + rp->r_dayofmonth - 1);
		else
		  result += sprintf(result, "J%d", total + rp->r_dayofmonth);
	} else {
		register int	week;
		register int	wday = rp->r_wday;
		register int	wdayoff;

		if (rp->r_dycode == DC_DOWGEQ) {
			wdayoff = (rp->r_dayofmonth - 1) % DAYSPERWEEK;
			if (wdayoff)
				compat = 2013;
			wday -= wdayoff;
			tod += wdayoff * SECSPERDAY;
			week = 1 + (rp->r_dayofmonth - 1) / DAYSPERWEEK;
		} else if (rp->r_dycode == DC_DOWLEQ) {
			if (rp->r_dayofmonth == len_months[1][rp->r_month])
				week = 5;
			else {
				wdayoff = rp->r_dayofmonth % DAYSPERWEEK;
				if (wdayoff)
					compat = 2013;
				wday -= wdayoff;
				tod += wdayoff * SECSPERDAY;
				week = rp->r_dayofmonth / DAYSPERWEEK;
			}
		} else	return -1;	/* "cannot happen" */
		if (wday < 0)
			wday += DAYSPERWEEK;
		result += sprintf(result, "M%d.%d.%d",
				  rp->r_month + 1, week, wday);
	}
	if (rp->r_todisut)
	  tod += stdoff;
	if (rp->r_todisstd && !rp->r_isdst)
	  tod += save;
	if (tod != 2 * SECSPERMIN * MINSPERHOUR) {
		*result++ = '/';
		if (! stringoffset(result, tod))
			return -1;
		if (tod < 0) {
			if (compat < 2013)
				compat = 2013;
		} else if (SECSPERDAY <= tod) {
			if (compat < 1994)
				compat = 1994;
		}
	}
	return compat;
}

static int
rule_cmp(struct rule const *a, struct rule const *b)
{
	if (!a)
		return -!!b;
	if (!b)
		return 1;
	if (a->r_hiyear != b->r_hiyear)
		return a->r_hiyear < b->r_hiyear ? -1 : 1;
	if (a->r_month - b->r_month != 0)
		return a->r_month - b->r_month;
	return a->r_dayofmonth - b->r_dayofmonth;
}

static int
stringzone(char *result, struct zone const *zpfirst, ptrdiff_t zonecount)
{
	register const struct zone *	zp;
	register struct rule *		rp;
	register struct rule *		stdrp;
	register struct rule *		dstrp;
	register ptrdiff_t		i;
	register const char *		abbrvar;
	register int			compat = 0;
	register int			c;
	size_t				len;
	int				offsetlen;
	struct rule			stdr, dstr;

	result[0] = '\0';

	/* Internet RFC 8536 section 5.1 says to use an empty TZ string if
	   future timestamps are truncated.  */
	if (hi_time < max_time)
	  return -1;

	zp = zpfirst + zonecount - 1;
	stdrp = dstrp = NULL;
	for (i = 0; i < zp->z_nrules; ++i) {
		rp = &zp->z_rules[i];
		if (rp->r_hiwasnum || rp->r_hiyear != ZIC_MAX)
			continue;
		if (rp->r_yrtype != NULL)
			continue;
		if (!rp->r_isdst) {
			if (stdrp == NULL)
				stdrp = rp;
			else	return -1;
		} else {
			if (dstrp == NULL)
				dstrp = rp;
			else	return -1;
		}
	}
	if (stdrp == NULL && dstrp == NULL) {
		/*
		** There are no rules running through "max".
		** Find the latest std rule in stdabbrrp
		** and latest rule of any type in stdrp.
		*/
		register struct rule *stdabbrrp = NULL;
		for (i = 0; i < zp->z_nrules; ++i) {
			rp = &zp->z_rules[i];
			if (!rp->r_isdst && rule_cmp(stdabbrrp, rp) < 0)
				stdabbrrp = rp;
			if (rule_cmp(stdrp, rp) < 0)
				stdrp = rp;
		}
		if (stdrp != NULL && stdrp->r_isdst) {
			/* Perpetual DST.  */
			dstr.r_month = TM_JANUARY;
			dstr.r_dycode = DC_DOM;
			dstr.r_dayofmonth = 1;
			dstr.r_tod = 0;
			dstr.r_todisstd = dstr.r_todisut = false;
			dstr.r_isdst = stdrp->r_isdst;
			dstr.r_save = stdrp->r_save;
			dstr.r_abbrvar = stdrp->r_abbrvar;
			stdr.r_month = TM_DECEMBER;
			stdr.r_dycode = DC_DOM;
			stdr.r_dayofmonth = 31;
			stdr.r_tod = SECSPERDAY + stdrp->r_save;
			stdr.r_todisstd = stdr.r_todisut = false;
			stdr.r_isdst = false;
			stdr.r_save = 0;
			stdr.r_abbrvar
			  = (stdabbrrp ? stdabbrrp->r_abbrvar : "");
			dstrp = &dstr;
			stdrp = &stdr;
		}
	}
	if (stdrp == NULL && (zp->z_nrules != 0 || zp->z_isdst))
		return -1;
	abbrvar = (stdrp == NULL) ? "" : stdrp->r_abbrvar;
	len = doabbr(result, zp, abbrvar, false, 0, true);
	offsetlen = stringoffset(result + len, - zp->z_stdoff);
	if (! offsetlen) {
		result[0] = '\0';
		return -1;
	}
	len += offsetlen;
	if (dstrp == NULL)
		return compat;
	len += doabbr(result + len, zp, dstrp->r_abbrvar,
		      dstrp->r_isdst, dstrp->r_save, true);
	if (dstrp->r_save != SECSPERMIN * MINSPERHOUR) {
	  offsetlen = stringoffset(result + len,
				   - (zp->z_stdoff + dstrp->r_save));
	  if (! offsetlen) {
	    result[0] = '\0';
	    return -1;
	  }
	  len += offsetlen;
	}
	result[len++] = ',';
	c = stringrule(result + len, dstrp, dstrp->r_save, zp->z_stdoff);
	if (c < 0) {
		result[0] = '\0';
		return -1;
	}
	if (compat < c)
		compat = c;
	len += strlen(result + len);
	result[len++] = ',';
	c = stringrule(result + len, stdrp, dstrp->r_save, zp->z_stdoff);
	if (c < 0) {
		result[0] = '\0';
		return -1;
	}
	if (compat < c)
		compat = c;
	return compat;
}

static void
outzone(const struct zone *zpfirst, ptrdiff_t zonecount)
{
	register const struct zone *	zp;
	register struct rule *		rp;
	register ptrdiff_t		i, j;
	register bool			usestart, useuntil;
	register zic_t			starttime, untiltime;
	register zic_t			stdoff;
	register zic_t			save;
	register zic_t			year;
	register zic_t			startoff;
	register bool			startttisstd;
	register bool			startttisut;
	register int			type;
	register char *			startbuf;
	register char *			ab;
	register char *			envvar;
	register int			max_abbr_len;
	register int			max_envvar_len;
	register bool			prodstic; /* all rules are min to max */
	register int			compat;
	register bool			do_extend;
	register char			version;
	ptrdiff_t lastatmax = -1;
	zic_t one = 1;
	zic_t y2038_boundary = one << 31;
	zic_t max_year0;
	int defaulttype = -1;

	max_abbr_len = 2 + max_format_len + max_abbrvar_len;
	max_envvar_len = 2 * max_abbr_len + 5 * 9;
	startbuf = emalloc(max_abbr_len + 1);
	ab = emalloc(max_abbr_len + 1);
	envvar = emalloc(max_envvar_len + 1);
	INITIALIZE(untiltime);
	INITIALIZE(starttime);
	/*
	** Now. . .finally. . .generate some useful data!
	*/
	timecnt = 0;
	typecnt = 0;
	charcnt = 0;
	prodstic = zonecount == 1;
	/*
	** Thanks to Earl Chew
	** for noting the need to unconditionally initialize startttisstd.
	*/
	startttisstd = false;
	startttisut = false;
	min_year = max_year = EPOCH_YEAR;
	if (leapseen) {
		updateminmax(leapminyear);
		updateminmax(leapmaxyear + (leapmaxyear < ZIC_MAX));
	}
	for (i = 0; i < zonecount; ++i) {
		zp = &zpfirst[i];
		if (i < zonecount - 1)
			updateminmax(zp->z_untilrule.r_loyear);
		for (j = 0; j < zp->z_nrules; ++j) {
			rp = &zp->z_rules[j];
			if (rp->r_lowasnum)
				updateminmax(rp->r_loyear);
			if (rp->r_hiwasnum)
				updateminmax(rp->r_hiyear);
			if (rp->r_lowasnum || rp->r_hiwasnum)
				prodstic = false;
		}
	}
	/*
	** Generate lots of data if a rule can't cover all future times.
	*/
	compat = stringzone(envvar, zpfirst, zonecount);
	version = compat < 2013 ? ZIC_VERSION_PRE_2013 : ZIC_VERSION;
	do_extend = compat < 0;
	if (noise) {
		if (!*envvar)
			warning("%s %s",
				_("no POSIX environment variable for zone"),
				zpfirst->z_name);
		else if (compat != 0) {
			/* Circa-COMPAT clients, and earlier clients, might
			   not work for this zone when given dates before
			   1970 or after 2038.  */
			warning(_("%s: pre-%d clients may mishandle"
				  " distant timestamps"),
				zpfirst->z_name, compat);
		}
	}
	if (do_extend) {
		/*
		** Search through a couple of extra years past the obvious
		** 400, to avoid edge cases.  For example, suppose a non-POSIX
		** rule applies from 2012 onwards and has transitions in March
		** and September, plus some one-off transitions in November
		** 2013.  If zic looked only at the last 400 years, it would
		** set max_year=2413, with the intent that the 400 years 2014
		** through 2413 will be repeated.  The last transition listed
		** in the tzfile would be in 2413-09, less than 400 years
		** after the last one-off transition in 2013-11.  Two years
		** might be overkill, but with the kind of edge cases
		** available we're not sure that one year would suffice.
		*/
		enum { years_of_observations = YEARSPERREPEAT + 2 };

		if (min_year >= ZIC_MIN + years_of_observations)
			min_year -= years_of_observations;
		else	min_year = ZIC_MIN;
		if (max_year <= ZIC_MAX - years_of_observations)
			max_year += years_of_observations;
		else	max_year = ZIC_MAX;
		/*
		** Regardless of any of the above,
		** for a "proDSTic" zone which specifies that its rules
		** always have and always will be in effect,
		** we only need one cycle to define the zone.
		*/
		if (prodstic) {
			min_year = 1900;
			max_year = min_year + years_of_observations;
		}
	}
	max_year0 = max_year;
	if (want_bloat()) {
	  /* For the benefit of older systems,
	     generate data from 1900 through 2038.  */
	  if (min_year > 1900)
		min_year = 1900;
	  if (max_year < 2038)
		max_year = 2038;
	}

	for (i = 0; i < zonecount; ++i) {
		struct rule *prevrp = NULL;
		/*
		** A guess that may well be corrected later.
		*/
		save = 0;
		zp = &zpfirst[i];
		usestart = i > 0 && (zp - 1)->z_untiltime > min_time;
		useuntil = i < (zonecount - 1);
		if (useuntil && zp->z_untiltime <= min_time)
			continue;
		stdoff = zp->z_stdoff;
		eat(zp->z_filename, zp->z_linenum);
		*startbuf = '\0';
		startoff = zp->z_stdoff;
		if (zp->z_nrules == 0) {
			save = zp->z_save;
			doabbr(startbuf, zp, NULL, zp->z_isdst, save, false);
			type = addtype(oadd(zp->z_stdoff, save),
				startbuf, zp->z_isdst, startttisstd,
				startttisut);
			if (usestart) {
				addtt(starttime, type);
				usestart = false;
			} else
				defaulttype = type;
		} else for (year = min_year; year <= max_year; ++year) {
			if (useuntil && year > zp->z_untilrule.r_hiyear)
				break;
			/*
			** Mark which rules to do in the current year.
			** For those to do, calculate rpytime(rp, year);
			*/
			for (j = 0; j < zp->z_nrules; ++j) {
				rp = &zp->z_rules[j];
				eats(zp->z_filename, zp->z_linenum,
					rp->r_filename, rp->r_linenum);
				rp->r_todo = year >= rp->r_loyear &&
						year <= rp->r_hiyear &&
						yearistype(year, rp->r_yrtype);
				if (rp->r_todo) {
					rp->r_temp = rpytime(rp, year);
					rp->r_todo
					  = (rp->r_temp < y2038_boundary
					     || year <= max_year0);
				}
			}
			for ( ; ; ) {
				register ptrdiff_t k;
				register zic_t	jtime, ktime;
				register zic_t	offset;

				INITIALIZE(ktime);
				if (useuntil) {
					/*
					** Turn untiltime into UT
					** assuming the current stdoff and
					** save values.
					*/
					untiltime = zp->z_untiltime;
					if (!zp->z_untilrule.r_todisut)
						untiltime = tadd(untiltime,
								 -stdoff);
					if (!zp->z_untilrule.r_todisstd)
						untiltime = tadd(untiltime,
								 -save);
				}
				/*
				** Find the rule (of those to do, if any)
				** that takes effect earliest in the year.
				*/
				k = -1;
				for (j = 0; j < zp->z_nrules; ++j) {
					rp = &zp->z_rules[j];
					if (!rp->r_todo)
						continue;
					eats(zp->z_filename, zp->z_linenum,
						rp->r_filename, rp->r_linenum);
					offset = rp->r_todisut ? 0 : stdoff;
					if (!rp->r_todisstd)
						offset = oadd(offset, save);
					jtime = rp->r_temp;
					if (jtime == min_time ||
						jtime == max_time)
							continue;
					jtime = tadd(jtime, -offset);
					if (k < 0 || jtime < ktime) {
						k = j;
						ktime = jtime;
					} else if (jtime == ktime) {
					  char const *dup_rules_msg =
					    _("two rules for same instant");
					  eats(zp->z_filename, zp->z_linenum,
					       rp->r_filename, rp->r_linenum);
					  warning("%s", dup_rules_msg);
					  rp = &zp->z_rules[k];
					  eats(zp->z_filename, zp->z_linenum,
					       rp->r_filename, rp->r_linenum);
					  error("%s", dup_rules_msg);
					}
				}
				if (k < 0)
					break;	/* go on to next year */
				rp = &zp->z_rules[k];
				rp->r_todo = false;
				if (useuntil && ktime >= untiltime)
					break;
				save = rp->r_save;
				if (usestart && ktime == starttime)
					usestart = false;
				if (usestart) {
					if (ktime < starttime) {
						startoff = oadd(zp->z_stdoff,
								save);
						doabbr(startbuf, zp,
							rp->r_abbrvar,
							rp->r_isdst,
							rp->r_save,
							false);
						continue;
					}
					if (*startbuf == '\0'
					    && startoff == oadd(zp->z_stdoff,
								save)) {
							doabbr(startbuf,
								zp,
								rp->r_abbrvar,
								rp->r_isdst,
								rp->r_save,
								false);
					}
				}
				eats(zp->z_filename, zp->z_linenum,
					rp->r_filename, rp->r_linenum);
				doabbr(ab, zp, rp->r_abbrvar,
				       rp->r_isdst, rp->r_save, false);
				offset = oadd(zp->z_stdoff, rp->r_save);
				if (!want_bloat() && !useuntil && !do_extend
				    && prevrp
				    && rp->r_hiyear == ZIC_MAX
				    && prevrp->r_hiyear == ZIC_MAX)
				  break;
				type = addtype(offset, ab, rp->r_isdst,
					rp->r_todisstd, rp->r_todisut);
				if (defaulttype < 0 && !rp->r_isdst)
				  defaulttype = type;
				if (rp->r_hiyear == ZIC_MAX
				    && ! (0 <= lastatmax
					  && ktime < attypes[lastatmax].at))
				  lastatmax = timecnt;
				addtt(ktime, type);
				prevrp = rp;
			}
		}
		if (usestart) {
			if (*startbuf == '\0' &&
				zp->z_format != NULL &&
				strchr(zp->z_format, '%') == NULL &&
				strchr(zp->z_format, '/') == NULL)
					strcpy(startbuf, zp->z_format);
			eat(zp->z_filename, zp->z_linenum);
			if (*startbuf == '\0')
error(_("can't determine time zone abbreviation to use just after until time"));
			else {
			  bool isdst = startoff != zp->z_stdoff;
			  type = addtype(startoff, startbuf, isdst,
					 startttisstd, startttisut);
			  if (defaulttype < 0 && !isdst)
			    defaulttype = type;
			  addtt(starttime, type);
			}
		}
		/*
		** Now we may get to set starttime for the next zone line.
		*/
		if (useuntil) {
			startttisstd = zp->z_untilrule.r_todisstd;
			startttisut = zp->z_untilrule.r_todisut;
			starttime = zp->z_untiltime;
			if (!startttisstd)
			  starttime = tadd(starttime, -save);
			if (!startttisut)
			  starttime = tadd(starttime, -stdoff);
		}
	}
	if (defaulttype < 0)
	  defaulttype = 0;
	if (0 <= lastatmax)
	  attypes[lastatmax].dontmerge = true;
	if (do_extend) {
		/*
		** If we're extending the explicitly listed observations
		** for 400 years because we can't fill the POSIX-TZ field,
		** check whether we actually ended up explicitly listing
		** observations through that period.  If there aren't any
		** near the end of the 400-year period, add a redundant
		** one at the end of the final year, to make it clear
		** that we are claiming to have definite knowledge of
		** the lack of transitions up to that point.
		*/
		struct rule xr;
		struct attype *lastat;
		xr.r_month = TM_JANUARY;
		xr.r_dycode = DC_DOM;
		xr.r_dayofmonth = 1;
		xr.r_tod = 0;
		for (lastat = attypes, i = 1; i < timecnt; i++)
			if (attypes[i].at > lastat->at)
				lastat = &attypes[i];
		if (!lastat || lastat->at < rpytime(&xr, max_year - 1)) {
			addtt(rpytime(&xr, max_year + 1),
			      lastat ? lastat->type : defaulttype);
			attypes[timecnt - 1].dontmerge = true;
		}
	}
	writezone(zpfirst->z_name, envvar, version, defaulttype);
	free(startbuf);
	free(ab);
	free(envvar);
}

static void
addtt(zic_t starttime, int type)
{
	attypes = growalloc(attypes, sizeof *attypes, timecnt, &timecnt_alloc);
	attypes[timecnt].at = starttime;
	attypes[timecnt].dontmerge = false;
	attypes[timecnt].type = type;
	++timecnt;
}

static int
addtype(zic_t utoff, char const *abbr, bool isdst, bool ttisstd, bool ttisut)
{
	register int	i, j;

	if (! (-1L - 2147483647L <= utoff && utoff <= 2147483647L)) {
		error(_("UT offset out of range"));
		exit(EXIT_FAILURE);
	}
	if (!want_bloat())
	  ttisstd = ttisut = false;

	for (j = 0; j < charcnt; ++j)
		if (strcmp(&chars[j], abbr) == 0)
			break;
	if (j == charcnt)
		newabbr(abbr);
	else {
	  /* If there's already an entry, return its index.  */
	  for (i = 0; i < typecnt; i++)
	    if (utoff == utoffs[i] && isdst == isdsts[i] && j == desigidx[i]
		&& ttisstd == ttisstds[i] && ttisut == ttisuts[i])
	      return i;
	}
	/*
	** There isn't one; add a new one, unless there are already too
	** many.
	*/
	if (typecnt >= TZ_MAX_TYPES) {
		error(_("too many local time types"));
		exit(EXIT_FAILURE);
	}
	i = typecnt++;
	utoffs[i] = utoff;
	isdsts[i] = isdst;
	ttisstds[i] = ttisstd;
	ttisuts[i] = ttisut;
	desigidx[i] = j;
	return i;
}

static void
leapadd(zic_t t, int correction, int rolling)
{
	register int i;

	if (TZ_MAX_LEAPS <= leapcnt) {
		error(_("too many leap seconds"));
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < leapcnt; ++i)
		if (t <= trans[i])
			break;
	memmove(&trans[i + 1], &trans[i], (leapcnt - i) * sizeof *trans);
	memmove(&corr[i + 1], &corr[i], (leapcnt - i) * sizeof *corr);
	memmove(&roll[i + 1], &roll[i], (leapcnt - i) * sizeof *roll);
	trans[i] = t;
	corr[i] = correction;
	roll[i] = rolling;
	++leapcnt;
}

static void
adjleap(void)
{
	register int	i;
	register zic_t	last = 0;
	register zic_t	prevtrans = 0;

	/*
	** propagate leap seconds forward
	*/
	for (i = 0; i < leapcnt; ++i) {
		if (trans[i] - prevtrans < 28 * SECSPERDAY) {
		  error(_("Leap seconds too close together"));
		  exit(EXIT_FAILURE);
		}
		prevtrans = trans[i];
		trans[i] = tadd(trans[i], last);
		last = corr[i] += last;
	}

	if (leapexpires < 0) {
	  leapexpires = comment_leapexpires;
	  if (0 <= leapexpires)
	    warning(_("\"#expires\" is obsolescent; use \"Expires\""));
	}

	if (0 <= leapexpires) {
	  leapexpires = oadd(leapexpires, last);
	  if (! (leapcnt == 0 || (trans[leapcnt - 1] < leapexpires))) {
	    error(_("last Leap time does not precede Expires time"));
	    exit(EXIT_FAILURE);
	  }
	  if (leapexpires <= hi_time)
	    hi_time = leapexpires - 1;
	}
}

static char *
shellquote(char *b, char const *s)
{
  *b++ = '\'';
  while (*s) {
    if (*s == '\'')
      *b++ = '\'', *b++ = '\\', *b++ = '\'';
    *b++ = *s++;
  }
  *b++ = '\'';
  return b;
}

static bool
yearistype(zic_t year, const char *type)
{
	char *buf;
	char *b;
	int result;

	if (type == NULL || *type == '\0')
		return true;
	buf = emalloc(1 + 4 * strlen(yitcommand) + 2
		      + INT_STRLEN_MAXIMUM(zic_t) + 2 + 4 * strlen(type) + 2);
	b = shellquote(buf, yitcommand);
	*b++ = ' ';
	b += sprintf(b, "%"PRIdZIC, year);
	*b++ = ' ';
	b = shellquote(b, type);
	*b = '\0';
	result = system(buf);
	if (WIFEXITED(result)) {
	  int status = WEXITSTATUS(result);
	  if (status <= 1) {
	    free(buf);
	    return status == 0;
	  }
	}
	error(_("Wild result from command execution"));
	fprintf(stderr, _("%s: command was '%s', result was %d\n"),
		progname, buf, result);
	exit(EXIT_FAILURE);
}

/* Is A a space character in the C locale?  */
static bool
is_space(char a)
{
	switch (a) {
	  default:
		return false;
	  case ' ': case '\f': case '\n': case '\r': case '\t': case '\v':
		return true;
	}
}

/* Is A an alphabetic character in the C locale?  */
static bool
is_alpha(char a)
{
	switch (a) {
	  default:
		return false;
	  case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
	  case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
	  case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
	  case 'V': case 'W': case 'X': case 'Y': case 'Z':
	  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
	  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
	  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
	  case 'v': case 'w': case 'x': case 'y': case 'z':
		return true;
	}
}

/* If A is an uppercase character in the C locale, return its lowercase
   counterpart.  Otherwise, return A.  */
static char
lowerit(char a)
{
	switch (a) {
	  default: return a;
	  case 'A': return 'a'; case 'B': return 'b'; case 'C': return 'c';
	  case 'D': return 'd'; case 'E': return 'e'; case 'F': return 'f';
	  case 'G': return 'g'; case 'H': return 'h'; case 'I': return 'i';
	  case 'J': return 'j'; case 'K': return 'k'; case 'L': return 'l';
	  case 'M': return 'm'; case 'N': return 'n'; case 'O': return 'o';
	  case 'P': return 'p'; case 'Q': return 'q'; case 'R': return 'r';
	  case 'S': return 's'; case 'T': return 't'; case 'U': return 'u';
	  case 'V': return 'v'; case 'W': return 'w'; case 'X': return 'x';
	  case 'Y': return 'y'; case 'Z': return 'z';
	}
}

/* case-insensitive equality */
static ATTRIBUTE_PURE bool
ciequal(register const char *ap, register const char *bp)
{
	while (lowerit(*ap) == lowerit(*bp++))
		if (*ap++ == '\0')
			return true;
	return false;
}

static ATTRIBUTE_PURE bool
itsabbr(register const char *abbr, register const char *word)
{
	if (lowerit(*abbr) != lowerit(*word))
		return false;
	++word;
	while (*++abbr != '\0')
		do {
			if (*word == '\0')
				return false;
		} while (lowerit(*word++) != lowerit(*abbr));
	return true;
}

/* Return true if ABBR is an initial prefix of WORD, ignoring ASCII case.  */

static ATTRIBUTE_PURE bool
ciprefix(char const *abbr, char const *word)
{
  do
    if (!*abbr)
      return true;
  while (lowerit(*abbr++) == lowerit(*word++));

  return false;
}

static const struct lookup *
byword(const char *word, const struct lookup *table)
{
	register const struct lookup *	foundlp;
	register const struct lookup *	lp;

	if (word == NULL || table == NULL)
		return NULL;

	/* If TABLE is LASTS and the word starts with "last" followed
	   by a non-'-', skip the "last" and look in WDAY_NAMES instead.
	   Warn about any usage of the undocumented prefix "last-".  */
	if (table == lasts && ciprefix("last", word) && word[4]) {
	  if (word[4] == '-')
	    warning(_("\"%s\" is undocumented; use \"last%s\" instead"),
		    word, word + 5);
	  else {
	    word += 4;
	    table = wday_names;
	  }
	}

	/*
	** Look for exact match.
	*/
	for (lp = table; lp->l_word != NULL; ++lp)
		if (ciequal(word, lp->l_word))
			return lp;
	/*
	** Look for inexact match.
	*/
	foundlp = NULL;
	for (lp = table; lp->l_word != NULL; ++lp)
		if (ciprefix(word, lp->l_word)) {
			if (foundlp == NULL)
				foundlp = lp;
			else	return NULL;	/* multiple inexact matches */
		}

	if (foundlp && noise) {
	  /* Warn about any backward-compatibility issue with pre-2017c zic.  */
	  bool pre_2017c_match = false;
	  for (lp = table; lp->l_word; lp++)
	    if (itsabbr(word, lp->l_word)) {
	      if (pre_2017c_match) {
		warning(_("\"%s\" is ambiguous in pre-2017c zic"), word);
		break;
	      }
	      pre_2017c_match = true;
	    }
	}

	return foundlp;
}

static char **
getfields(register char *cp)
{
	register char *		dp;
	register char **	array;
	register int		nsubs;

	if (cp == NULL)
		return NULL;
	array = emalloc(size_product(strlen(cp) + 1, sizeof *array));
	nsubs = 0;
	for ( ; ; ) {
		while (is_space(*cp))
				++cp;
		if (*cp == '\0' || *cp == '#')
			break;
		array[nsubs++] = dp = cp;
		do {
			if ((*dp = *cp++) != '"')
				++dp;
			else while ((*dp = *cp++) != '"')
				if (*dp != '\0')
					++dp;
				else {
				  error(_("Odd number of quotation marks"));
				  exit(EXIT_FAILURE);
				}
		} while (*cp && *cp != '#' && !is_space(*cp));
		if (is_space(*cp))
			++cp;
		*dp = '\0';
	}
	array[nsubs] = NULL;
	return array;
}

static _Noreturn void
time_overflow(void)
{
  error(_("time overflow"));
  exit(EXIT_FAILURE);
}

static ATTRIBUTE_PURE zic_t
oadd(zic_t t1, zic_t t2)
{
	if (t1 < 0 ? t2 < ZIC_MIN - t1 : ZIC_MAX - t1 < t2)
	  time_overflow();
	return t1 + t2;
}

static ATTRIBUTE_PURE zic_t
tadd(zic_t t1, zic_t t2)
{
  if (t1 < 0) {
    if (t2 < min_time - t1) {
      if (t1 != min_time)
	time_overflow();
      return min_time;
    }
  } else {
    if (max_time - t1 < t2) {
      if (t1 != max_time)
	time_overflow();
      return max_time;
    }
  }
  return t1 + t2;
}

/*
** Given a rule, and a year, compute the date (in seconds since January 1,
** 1970, 00:00 LOCAL time) in that year that the rule refers to.
*/

static zic_t
rpytime(const struct rule *rp, zic_t wantedy)
{
	register int	m, i;
	register zic_t	dayoff;			/* with a nod to Margaret O. */
	register zic_t	t, y;

	if (wantedy == ZIC_MIN)
		return min_time;
	if (wantedy == ZIC_MAX)
		return max_time;
	dayoff = 0;
	m = TM_JANUARY;
	y = EPOCH_YEAR;
	if (y < wantedy) {
	  wantedy -= y;
	  dayoff = (wantedy / YEARSPERREPEAT) * (SECSPERREPEAT / SECSPERDAY);
	  wantedy %= YEARSPERREPEAT;
	  wantedy += y;
	} else if (wantedy < 0) {
	  dayoff = (wantedy / YEARSPERREPEAT) * (SECSPERREPEAT / SECSPERDAY);
	  wantedy %= YEARSPERREPEAT;
	}
	while (wantedy != y) {
		if (wantedy > y) {
			i = len_years[isleap(y)];
			++y;
		} else {
			--y;
			i = -len_years[isleap(y)];
		}
		dayoff = oadd(dayoff, i);
	}
	while (m != rp->r_month) {
		i = len_months[isleap(y)][m];
		dayoff = oadd(dayoff, i);
		++m;
	}
	i = rp->r_dayofmonth;
	if (m == TM_FEBRUARY && i == 29 && !isleap(y)) {
		if (rp->r_dycode == DC_DOWLEQ)
			--i;
		else {
			error(_("use of 2/29 in non leap-year"));
			exit(EXIT_FAILURE);
		}
	}
	--i;
	dayoff = oadd(dayoff, i);
	if (rp->r_dycode == DC_DOWGEQ || rp->r_dycode == DC_DOWLEQ) {
		register zic_t	wday;

#define LDAYSPERWEEK	((zic_t) DAYSPERWEEK)
		wday = EPOCH_WDAY;
		/*
		** Don't trust mod of negative numbers.
		*/
		if (dayoff >= 0)
			wday = (wday + dayoff) % LDAYSPERWEEK;
		else {
			wday -= ((-dayoff) % LDAYSPERWEEK);
			if (wday < 0)
				wday += LDAYSPERWEEK;
		}
		while (wday != rp->r_wday)
			if (rp->r_dycode == DC_DOWGEQ) {
				dayoff = oadd(dayoff, 1);
				if (++wday >= LDAYSPERWEEK)
					wday = 0;
				++i;
			} else {
				dayoff = oadd(dayoff, -1);
				if (--wday < 0)
					wday = LDAYSPERWEEK - 1;
				--i;
			}
		if (i < 0 || i >= len_months[isleap(y)][m]) {
			if (noise)
				warning(_("rule goes past start/end of month; \
will not work with pre-2004 versions of zic"));
		}
	}
	if (dayoff < min_time / SECSPERDAY)
		return min_time;
	if (dayoff > max_time / SECSPERDAY)
		return max_time;
	t = (zic_t) dayoff * SECSPERDAY;
	return tadd(t, rp->r_tod);
}

static void
newabbr(const char *string)
{
	register int	i;

	if (strcmp(string, GRANDPARENTED) != 0) {
		register const char *	cp;
		const char *		mp;

		cp = string;
		mp = NULL;
		while (is_alpha(*cp) || ('0' <= *cp && *cp <= '9')
		       || *cp == '-' || *cp == '+')
				++cp;
		if (noise && cp - string < 3)
		  mp = _("time zone abbreviation has fewer than 3 characters");
		if (cp - string > ZIC_MAX_ABBR_LEN_WO_WARN)
		  mp = _("time zone abbreviation has too many characters");
		if (*cp != '\0')
mp = _("time zone abbreviation differs from POSIX standard");
		if (mp != NULL)
			warning("%s (%s)", mp, string);
	}
	i = strlen(string) + 1;
	if (charcnt + i > TZ_MAX_CHARS) {
		error(_("too many, or too long, time zone abbreviations"));
		exit(EXIT_FAILURE);
	}
	strcpy(&chars[charcnt], string);
	charcnt += i;
}

/* Ensure that the directories of ARGNAME exist, by making any missing
   ones.  If ANCESTORS, do this only for ARGNAME's ancestors; otherwise,
   do it for ARGNAME too.  Exit with failure if there is trouble.
   Do not consider an existing non-directory to be trouble.  */
static void
mkdirs(char const *argname, bool ancestors)
{
	register char *	name;
	register char *	cp;

	cp = name = ecpyalloc(argname);

	/* On MS-Windows systems, do not worry about drive letters or
	   backslashes, as this should suffice in practice.  Time zone
	   names do not use drive letters and backslashes.  If the -d
	   option of zic does not name an already-existing directory,
	   it can use slashes to separate the already-existing
	   ancestor prefix from the to-be-created subdirectories.  */

	/* Do not mkdir a root directory, as it must exist.  */
	while (*cp == '/')
	  cp++;

	while (cp && ((cp = strchr(cp, '/')) || !ancestors)) {
		if (cp)
		  *cp = '\0';
		/*
		** Try to create it.  It's OK if creation fails because
		** the directory already exists, perhaps because some
		** other process just created it.  For simplicity do
		** not check first whether it already exists, as that
		** is checked anyway if the mkdir fails.
		*/
		if (mkdir(name, MKDIR_UMASK) != 0) {
			/* For speed, skip itsdir if errno == EEXIST.  Since
			   mkdirs is called only after open fails with ENOENT
			   on a subfile, EEXIST implies itsdir here.  */
			int err = errno;
			if (err != EEXIST && !itsdir(name)) {
				error(_("%s: Can't create directory %s: %s"),
				      progname, name, strerror(err));
				exit(EXIT_FAILURE);
			}
		}
		if (cp)
		  *cp++ = '/';
	}
	free(name);
}
