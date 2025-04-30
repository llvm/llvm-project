/* Deallocation thread-specific data structures related to pthread_key_create.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthreadP.h>

/* Deallocate POSIX thread-local-storage.  */
void
__nptl_deallocate_tsd (void)
{
  struct pthread *self = THREAD_SELF;

  /* Maybe no data was ever allocated.  This happens often so we have
     a flag for this.  */
  if (THREAD_GETMEM (self, specific_used))
    {
      size_t round;
      size_t cnt;

      round = 0;
      do
        {
          size_t idx;

          /* So far no new nonzero data entry.  */
          THREAD_SETMEM (self, specific_used, false);

          for (cnt = idx = 0; cnt < PTHREAD_KEY_1STLEVEL_SIZE; ++cnt)
            {
              struct pthread_key_data *level2;

              level2 = THREAD_GETMEM_NC (self, specific, cnt);

              if (level2 != NULL)
                {
                  size_t inner;

                  for (inner = 0; inner < PTHREAD_KEY_2NDLEVEL_SIZE;
                       ++inner, ++idx)
                    {
                      void *data = level2[inner].data;

                      if (data != NULL)
                        {
                          /* Always clear the data.  */
                          level2[inner].data = NULL;

                          /* Make sure the data corresponds to a valid
                             key.  This test fails if the key was
                             deallocated and also if it was
                             re-allocated.  It is the user's
                             responsibility to free the memory in this
                             case.  */
                          if (level2[inner].seq
                              == __pthread_keys[idx].seq
                              /* It is not necessary to register a destructor
                                 function.  */
                              && __pthread_keys[idx].destr != NULL)
                            /* Call the user-provided destructor.  */
                            __pthread_keys[idx].destr (data);
                        }
                    }
                }
              else
                idx += PTHREAD_KEY_1STLEVEL_SIZE;
            }

          if (THREAD_GETMEM (self, specific_used) == 0)
            /* No data has been modified.  */
            goto just_free;
        }
      /* We only repeat the process a fixed number of times.  */
      while (__builtin_expect (++round < PTHREAD_DESTRUCTOR_ITERATIONS, 0));

      /* Just clear the memory of the first block for reuse.  */
      memset (&THREAD_SELF->specific_1stblock, '\0',
              sizeof (self->specific_1stblock));

    just_free:
      /* Free the memory for the other blocks.  */
      for (cnt = 1; cnt < PTHREAD_KEY_1STLEVEL_SIZE; ++cnt)
        {
          struct pthread_key_data *level2;

          level2 = THREAD_GETMEM_NC (self, specific, cnt);
          if (level2 != NULL)
            {
              /* The first block is allocated as part of the thread
                 descriptor.  */
              free (level2);
              THREAD_SETMEM_NC (self, specific, cnt, NULL);
            }
        }

      THREAD_SETMEM (self, specific_used, false);
    }
}
libc_hidden_def (__nptl_deallocate_tsd)
