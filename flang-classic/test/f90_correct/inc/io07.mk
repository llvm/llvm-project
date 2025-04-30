#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io07  ########


io07: run
	

build:  $(SRC)/io07.f90
	-$(RM) io07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io07.f90 -o io07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io07.$(OBJX) check.$(OBJX) $(LIBS) -o io07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io07
	io07.$(EXESUFFIX)

verify: ;

