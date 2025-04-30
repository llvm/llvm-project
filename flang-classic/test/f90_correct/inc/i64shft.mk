#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test i64shft  ########


i64shft: run
	

build:  $(SRC)/i64shft.f90
	-$(RM) i64shft.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c -i8 $(FFLAGS) $(LDFLAGS) $(SRC)/i64shft.f90 -o i64shft.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) i64shft.$(OBJX) check.$(OBJX) $(LIBS) -o i64shft.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test i64shft
	i64shft.$(EXESUFFIX)

verify: ;

i64shft.run: run

