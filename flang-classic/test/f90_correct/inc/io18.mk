#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io18  ########


io18: run


build:  $(SRC)/io18.f90
	-$(RM) io18.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io18.f90 -o io18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io18.$(OBJX) check.$(OBJX) $(LIBS) -o io18.$(EXESUFFIX)


run:
	-$(CP) $(SRC)/io08.inp .
	@echo ------------------------------------ executing test io18
	io18.$(EXESUFFIX)

verify: ;
