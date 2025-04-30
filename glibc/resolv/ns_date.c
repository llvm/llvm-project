/*
 * Copyright (c) 2004 by Internet Systems Consortium, Inc. ("ISC")
 * Copyright (c) 1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS.  IN NO EVENT SHALL ISC BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/* Import. */

#include <arpa/nameser.h>

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SPRINTF(x) ((size_t)sprintf x)

/* Forward. */

static int	datepart(const char *, int, int, int, int *);

/* Public. */

/*%
 * Convert a date in ASCII into the number of seconds since
 * 1 January 1970 (GMT assumed).  Format is yyyymmddhhmmss, all
 * digits required, no spaces allowed.
 */

uint32_t
ns_datetosecs(const char *cp, int *errp) {
	struct tm time;
	uint32_t result;
	int mdays, i;
	static const int days_per_month[12] =
		{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

	if (strlen(cp) != 14U) {
		*errp = 1;
		return (0);
	}
	*errp = 0;

	memset(&time, 0, sizeof time);
	time.tm_year  = datepart(cp +  0, 4, 1990, 9999, errp) - 1900;
	time.tm_mon   = datepart(cp +  4, 2,   01,   12, errp) - 1;
	time.tm_mday  = datepart(cp +  6, 2,   01,   31, errp);
	time.tm_hour  = datepart(cp +  8, 2,   00,   23, errp);
	time.tm_min   = datepart(cp + 10, 2,   00,   59, errp);
	time.tm_sec   = datepart(cp + 12, 2,   00,   59, errp);
	if (*errp)		/*%< Any parse errors? */
		return (0);

	/*
	 * OK, now because timegm() is not available in all environments,
	 * we will do it by hand.  Roll up sleeves, curse the gods, begin!
	 */

#define SECS_PER_DAY    ((uint32_t)24*60*60)
#define isleap(y) ((((y) % 4) == 0 && ((y) % 100) != 0) || ((y) % 400) == 0)

	result  = time.tm_sec;				/*%< Seconds */
	result += time.tm_min * 60;			/*%< Minutes */
	result += time.tm_hour * (60*60);		/*%< Hours */
	result += (time.tm_mday - 1) * SECS_PER_DAY;	/*%< Days */
	/* Months are trickier.  Look without leaping, then leap */
	mdays = 0;
	for (i = 0; i < time.tm_mon; i++)
		mdays += days_per_month[i];
	result += mdays * SECS_PER_DAY;			/*%< Months */
	if (time.tm_mon > 1 && isleap(1900+time.tm_year))
		result += SECS_PER_DAY;		/*%< Add leapday for this year */
	/* First figure years without leapdays, then add them in.  */
	/* The loop is slow, FIXME, but simple and accurate.  */
	result += (time.tm_year - 70) * (SECS_PER_DAY*365); /*%< Years */
	for (i = 70; i < time.tm_year; i++)
		if (isleap(1900+i))
			result += SECS_PER_DAY; /*%< Add leapday for prev year */
	return (result);
}

/* Private. */

/*%
 * Parse part of a date.  Set error flag if any error.
 * Don't reset the flag if there is no error.
 */
static int
datepart(const char *buf, int size, int min, int max, int *errp) {
	int result = 0;
	int i;

	for (i = 0; i < size; i++) {
		if (!isdigit((unsigned char)(buf[i])))
			*errp = 1;
		result = (result * 10) + buf[i] - '0';
	}
	if (result < min)
		*errp = 1;
	if (result > max)
		*errp = 1;
	return (result);
}

/*! \file */
