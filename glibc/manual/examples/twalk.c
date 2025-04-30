/* Implement twalk using twalk_r.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.
*/

#include <search.h>

struct twalk_with_twalk_r_closure
{
  void (*action) (const void *, VISIT, int);
  int depth;
};

static void
twalk_with_twalk_r_action (const void *nodep, VISIT which, void *closure0)
{
  struct twalk_with_twalk_r_closure *closure = closure0;

  switch (which)
    {
    case leaf:
      closure->action (nodep, which, closure->depth);
      break;
    case preorder:
      closure->action (nodep, which, closure->depth);
      ++closure->depth;
      break;
    case postorder:
      /* The preorder action incremented the depth.  */
      closure->action (nodep, which, closure->depth - 1);
      break;
    case endorder:
      --closure->depth;
      closure->action (nodep, which, closure->depth);
      break;
    }
}

void
twalk (const void *root, void (*action) (const void *, VISIT, int))
{
  struct twalk_with_twalk_r_closure closure = { action, 0 };
  twalk_r (root, twalk_with_twalk_r_action, &closure);
}
