#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fn16  ########


fn16: fn16.$(OBJX)
	

fn16.$(OBJX):  $(SRC)/fn16.f
	-$(RM) fn16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fn16.f -o fn16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fn16.$(OBJX) check.$(OBJX) $(LIBS) -o fn16.$(EXESUFFIX)



fn16.run: fn16.$(OBJX)
	@echo ------------------------------------ executing test fn16
	fn16.$(EXESUFFIX)

run: fn16.$(OBJX)
	@echo ------------------------------------ executing test fn16
	fn16.$(EXESUFFIX)

verify: ;


build:	fn16.$(OBJX)

