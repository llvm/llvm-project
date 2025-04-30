#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qsqrt qexp qlog  ########


qsqrt_qexp_qlog: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qsqrt_qexp_qlog.f08 $(SRC)/qsqrt_qexp_qlog_expct.c fcheck.$(OBJX)
	-$(RM) qsqrt_qexp_qlog.$(EXESUFFIX) qsqrt_qexp_qlog.$(OBJX) qsqrt_qexp_qlog_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/qsqrt_qexp_qlog_expct.c -o qsqrt_qexp_qlog_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qsqrt_qexp_qlog.f08 -o qsqrt_qexp_qlog.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qsqrt_qexp_qlog.$(OBJX) qsqrt_qexp_qlog_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qsqrt_qexp_qlog.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qsqrt_qexp_qlog
	qsqrt_qexp_qlog.$(EXESUFFIX)

verify: ;

qsqrt_qexp_qlog.run: run

