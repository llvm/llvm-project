#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test io27  ########


io27: run


build:  $(SRC)/io27.f08
	-$(RM) io27.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/io27.f08 -o io27.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) io27.$(OBJX) check.$(OBJX) $(LIBS) -o io27.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test io27
	io27.$(EXESUFFIX)

verify: ;
