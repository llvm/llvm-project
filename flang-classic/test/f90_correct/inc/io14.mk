#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io14  ########


io14: run


build:  $(SRC)/io14.f90
	-$(RM) io14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io14.f90 -o io14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io14.$(OBJX) check.$(OBJX) $(LIBS) -o io14.$(EXESUFFIX)


run:
	-$(CP) $(SRC)/io08.inp .
	@echo ------------------------------------ executing test io14
	io14.$(EXESUFFIX)

verify: ;
