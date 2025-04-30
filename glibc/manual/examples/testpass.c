/* Verify a passphrase.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <crypt.h>

/* @samp{GNU's Not Unix} hashed using SHA-256, MD5, and DES.  */
static const char hash_sha[] =
  "$5$DQ2z5NHf1jNJnChB$kV3ZTR0aUaosujPhLzR84Llo3BsspNSe4/tsp7VoEn6";
static const char hash_md5[] = "$1$A3TxDv41$rtXVTUXl2LkeSV0UU5xxs1";
static const char hash_des[] = "FgkTuF98w5DaI";

int
main(void)
{
  char *phrase;
  int status = 0;

  /* Prompt for a passphrase.  */
  phrase = getpass ("Enter passphrase: ");

  /* Compare against the stored hashes.  Any input that begins with
     @samp{GNU's No} will match the DES hash, but the other two will
     only match @samp{GNU's Not Unix}.  */

  if (strcmp (crypt (phrase, hash_sha), hash_sha))
    {
      puts ("SHA: not ok");
      status = 1;
    }
  else
    puts ("SHA: ok");

  if (strcmp (crypt (phrase, hash_md5), hash_md5))
    {
      puts ("MD5: not ok");
      status = 1;
    }
  else
    puts ("MD5: ok");

  if (strcmp (crypt (phrase, hash_des), hash_des))
    {
      puts ("DES: not ok");
      status = 1;
    }
  else
    puts ("DES: ok");

  return status;
}
