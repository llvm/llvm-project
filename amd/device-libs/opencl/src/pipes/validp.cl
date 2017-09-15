/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/


__attribute__((overloadable, always_inline)) bool
is_valid_reserve_id(reserve_id_t rid)
{
    return as_ulong(rid) != ~(size_t)0;
}

