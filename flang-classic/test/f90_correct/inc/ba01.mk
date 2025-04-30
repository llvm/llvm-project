#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ba01  ########


ba01: run
	

build:  $(SRC)/ba01.f
	-$(RM) ba01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ba01.f -o ba01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ba01.$(OBJX) check.$(OBJX) $(LIBS) -o ba01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ba01
	ba01.$(EXESUFFIX)

verify: ;

ba01.run: run

