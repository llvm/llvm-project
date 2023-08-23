/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

void
__memcpy_internal_aligned(void *d, const void *s, size_t size, size_t align)
{
    if (align == 2) {
	short *d2 = (short *)d;
	short *s2 = (short *)s;
	short *e2 = s2 + size/2;

	while (s2 < e2)
	    *d2++ = *s2++;
    } else if (align == 4) {
	int *d4 = (int *)d;
	int *s4 = (int *)s;
	int *e4 = s4 + size/4;

	while (s4 < e4)
	    *d4++ = *s4++;
    } else if (align == 8) {
	long *d8 = (long *)d;
	long *s8 = (long *)s;
	long *e8 = s8 + size/8;

	while (s8 < e8)
	    *d8++ = *s8++;
    } else if (align == 16) {
	long2 *d16 = (long2 *)d;
	long2 *s16 = (long2 *)s;
	long2 *e16 = s16 + size/16;

	while (s16 < e16)
	    *d16++ = *s16++;
    } else if (align == 32 || align == 64 || align == 128) {
	long4 *d32 = (long4 *)d;
	long4 *s32 = (long4 *)s;
	long4 *e32 = s32 + size/32;

	while (s32 < e32)
	    *d32++ = *s32++;
    } else {
	char *d1 = (char *)d;
	char *s1 = (char *)s;
	char *e1 = s1 + size;

	while (s1 < e1)
	    *d1++ = *s1++;
    }
}

