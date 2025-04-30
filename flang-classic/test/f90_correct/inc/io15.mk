#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io15  ########


io15: run


build:  $(SRC)/io15.f90
	-$(RM) io15.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io15.f90 -o io15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io15.$(OBJX) check.$(OBJX) $(LIBS) -o io15.$(EXESUFFIX)


run:
	-$(CP) $(SRC)/io08.inp .
	@echo ------------------------------------ executing test io15
	io15.$(EXESUFFIX)

verify: ;
