/* Encrypting Passwords
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
#include <unistd.h>
#include <crypt.h>

int
main(void)
{
  unsigned char ubytes[16];
  char salt[20];
  const char *const saltchars =
    "./0123456789ABCDEFGHIJKLMNOPQRST"
    "UVWXYZabcdefghijklmnopqrstuvwxyz";
  char *hash;
  int i;

  /* Retrieve 16 unpredictable bytes from the operating system.  */
  if (getentropy (ubytes, sizeof ubytes))
    {
      perror ("getentropy");
      return 1;
    }

  /* Use them to fill in the salt string.  */
  salt[0] = '$';
  salt[1] = '5'; /* SHA-256 */
  salt[2] = '$';
  for (i = 0; i < 16; i++)
    salt[3+i] = saltchars[ubytes[i] & 0x3f];
  salt[3+i] = '\0';

  /* Read in the user's passphrase and hash it.  */
  hash = crypt (getpass ("Enter new passphrase: "), salt);
  if (!hash || hash[0] == '*')
    {
      perror ("crypt");
      return 1;
    }

  /* Print the results.  */
  puts (hash);
  return 0;
}
