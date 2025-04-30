#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp02  ########


qp02: run


build:  $(SRC)/qp02.f08
	-$(RM) qp02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp02.f08 -o qp02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp02.$(OBJX) check.$(OBJX) $(LIBS) -o qp02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp02
	qp02.$(EXESUFFIX)

verify: ;
