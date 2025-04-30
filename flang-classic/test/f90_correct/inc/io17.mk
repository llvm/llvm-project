#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io17  ########


io17: run


build:  $(SRC)/io17.f90
	-$(RM) io17.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io17.f90 -o io17.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io17.$(OBJX) check.$(OBJX) $(LIBS) -o io17.$(EXESUFFIX)


run:
	-$(CP) $(SRC)/io08.inp .
	@echo ------------------------------------ executing test io17
	io17.$(EXESUFFIX)

verify: ;
