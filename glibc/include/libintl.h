#ifndef _LIBINTL_H
#include <intl/libintl.h>

# ifndef _ISOMAC

#include <locale.h>

/* Now define the internal interfaces.  */
extern char *__gettext (const char *__msgid)
     __attribute_format_arg__ (1);
extern char *__dgettext (const char *__domainname,
			 const char *__msgid)
     __attribute_format_arg__ (2);
extern char *__dcgettext (const char *__domainname,
			  const char *__msgid, int __category)
     __attribute_format_arg__ (2);
libc_hidden_proto (__dcgettext)

extern char *__ngettext (const char *__msgid1, const char *__msgid2,
			 unsigned long int __n)
     __attribute_format_arg__ (1) __attribute_format_arg__ (2);
extern char *__dngettext (const char *__domainname,
			  const char *__msgid1, const char *__msgid2,
			  unsigned long int __n)
     __attribute_format_arg__ (2) __attribute_format_arg__ (3);
extern char *__dcngettext (const char *__domainname,
			   const char *__msgid1, const char *__msgid2,
			   unsigned long int __n, int __category)
     __attribute_format_arg__ (2) __attribute_format_arg__ (3);

extern char *__textdomain (const char *__domainname);
extern char *__bindtextdomain (const char *__domainname,
			       const char *__dirname);
extern char *__bind_textdomain_codeset (const char *__domainname,
					const char *__codeset);

extern const char _libc_intl_domainname[];
libc_hidden_proto (_libc_intl_domainname)

/* _ marks its argument, a string literal, for translation, and
   performs translation at run time if the LC_MESSAGES locale category
   has been set.  The MSGID argument is extracted, added to the
   translation database, and eventually submitted to the translation
   team for processing.  New translations are periodically
   incorporated into the glibc source tree as part of translation
   updates.  */
# undef _
# define _(msgid) __dcgettext (_libc_intl_domainname, msgid, LC_MESSAGES)

/* N_ marks its argument, a string literal, for translation, so that
   it is extracted and added to the translation database (similar to
   the _ macro above).  It does not translate the string at run time.
   The first, primary use case for N_ is a context in which a string
   literal is required, such as an initializer.  Translation will
   happen later, for example using the __gettext function.

   The second, historic, use case involves strings which may be
   translated in a future version of the library, but cannot be
   translated in current releases due to some technical limitation
   (e.g., gettext not being available in the dynamic loader).  No
   translation at run time happens in such cases.  In the future, this
   historic usage of N_ may become deprecated.  Strings which are not
   translated create unnecessary work for the translation team.  We
   continue to use N_ because it helps mark translatable strings.  */
# undef N_
# define N_(msgid)	msgid

# endif /* !_ISOMAC */
#endif
