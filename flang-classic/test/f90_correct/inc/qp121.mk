#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtos  ########


qp121: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp121.f08 fcheck.$(OBJX)
	-$(RM) qp121.$(EXESUFFIX) qp121.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp121.f08 -o qp121.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp121.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp121.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp121
	qp121.$(EXESUFFIX)

verify: ;

qp121.run: run

