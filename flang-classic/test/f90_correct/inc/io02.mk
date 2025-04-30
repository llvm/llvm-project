#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io02  ########


io02: run
	

build:  $(SRC)/io02.f90
	-$(RM) io02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io02.f90 -o io02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io02.$(OBJX) check.$(OBJX) $(LIBS) -o io02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io02
	io02.$(EXESUFFIX)

verify: ;

io02.run: run

