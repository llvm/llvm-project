
! KGEN-generated Fortran source file
!
! Filename    : perf_mod.F90
! Generated at: 2015-02-24 15:34:48
! KGEN version: 0.4.4



    MODULE perf_mod
        !-----------------------------------------------------------------------
        !
        ! Purpose: This module is responsible for controlling the performance
        !          timer logic.
        !
        ! Author:  P. Worley, January 2007
        !
        ! $Id$
        !
        !-----------------------------------------------------------------------
        !-----------------------------------------------------------------------
        !- Uses ----------------------------------------------------------------
        !-----------------------------------------------------------------------
        USE perf_utils, only : shr_kind_i8
        !-----------------------------------------------------------------------
        !- module boilerplate --------------------------------------------------
        !-----------------------------------------------------------------------
        IMPLICIT NONE
        PRIVATE ! Make the default access private
        !
        ! Copyright (C) 2003-2014 Intel Corporation.  All Rights Reserved.
        !
        ! The source code contained or described herein and all documents
        ! related to the source code ("Material") are owned by Intel Corporation
        ! or its suppliers or licensors.  Title to the Material remains with
        ! Intel Corporation or its suppliers and licensors.  The Material is
        ! protected by worldwide copyright and trade secret laws and treaty
        ! provisions.  No part of the Material may be used, copied, reproduced,
        ! modified, published, uploaded, posted, transmitted, distributed, or
        ! disclosed in any way without Intel's prior express written permission.
        !
        ! No license under any patent, copyright, trade secret or other
        ! intellectual property right is granted to or conferred upon you by
        ! disclosure or delivery of the Materials, either expressly, by
        ! implication, inducement, estoppel or otherwise.  Any license under
        ! such intellectual property rights must be express and approved by
        ! Intel in writing.
        !      /* -*- Mode: Fortran; -*- */
        !
        !      (C) 2001 by Argonne National Laboratory.
        !
        !                                 MPICH2 COPYRIGHT
        !
        ! The following is a notice of limited availability of the code, and disclaimer
        ! which must be included in the prologue of the code and in all source listings
        ! of the code.
        !
        ! Copyright Notice
        !  + 2002 University of Chicago
        !
        ! Permission is hereby granted to use, reproduce, prepare derivative works, and
        ! to redistribute to others.  This software was authored by:
        !
        ! Mathematics and Computer Science Division
        ! Argonne National Laboratory, Argonne IL 60439
        !
        ! (and)
        !
        ! Department of Computer Science
        ! University of Illinois at Urbana-Champaign
        !
        !
        !                             GOVERNMENT LICENSE
        !
        ! Portions of this material resulted from work developed under a U.S.
        ! Government Contract and are subject to the following license: the Government
        ! is granted for itself and others acting on its behalf a paid-up, nonexclusive,
        ! irrevocable worldwide license in this computer software to reproduce, prepare
        ! derivative works, and perform publicly and display publicly.
        !
        !                                 DISCLAIMER
        !
        ! This computer code material was prepared, in part, as an account of work
        ! sponsored by an agency of the United States Government.  Neither the United
        ! States, nor the University of Chicago, nor any of their employees, makes any
        ! warranty express or implied, or assumes any legal liability or responsibility
        ! for the accuracy, completeness, or usefulness of any information, apparatus,
        ! product, or process disclosed, or represents that its use would not infringe
        ! privately owned rights.
        !
        ! Portions of this code were written by Microsoft. Those portions are
        ! Copyright (c) 2007 Microsoft Corporation. Microsoft grants permission to
        ! use, reproduce, prepare derivative works, and to redistribute to
        ! others. The code is licensed "as is." The User bears the risk of using
        ! it. Microsoft gives no express warranties, guarantees or
        ! conditions. To the extent permitted by law, Microsoft excludes the
        ! implied warranties of merchantability, fitness for a particular
        ! purpose and non-infringement.
        !
        !
        !
        !
        !
        !      DO NOT EDIT
        !      This file created by buildiface
        !
        !-----------------------------------------------------------------------
        ! Public interfaces ----------------------------------------------------
        !-----------------------------------------------------------------------
        PUBLIC t_startf
        PUBLIC t_stopf
        !-----------------------------------------------------------------------
        ! Private interfaces (local) -------------------------------------------
        !-----------------------------------------------------------------------
        !-----------------------------------------------------------------------
        !- include statements --------------------------------------------------
        !-----------------------------------------------------------------------
        !
        ! $Id: gptl.inc,v 1.44 2011-03-28 20:55:19 rosinski Exp $
        !
        ! Author: Jim Rosinski
        !
        ! GPTL header file to be included in user code. Values match
        ! their counterparts in gptl.h. See that file or man pages
        ! or web-based documenation for descriptions of each value
        !
        ! Externals
        !-----------------------------------------------------------------------
        ! Private data ---------------------------------------------------------
        !-----------------------------------------------------------------------
        !----------------------------------------------------------------------------
        ! perf_mod options
        !----------------------------------------------------------------------------
        ! default
        ! unit number for log output
        LOGICAL, parameter :: def_timing_initialized = .false. ! default
        LOGICAL, private :: timing_initialized = def_timing_initialized
        ! flag indicating whether timing library has
        ! been initialized
        ! default
        ! flag indicating whether timers are disabled
        ! default
        ! flag indicating whether the mpi_barrier in
        ! t_barrierf should be called
        ! default
        ! integer indicating maximum number of levels of
        ! timer nesting
        INTEGER, parameter :: def_timing_detail_limit = 1 ! default
        INTEGER, private :: timing_detail_limit = def_timing_detail_limit
        ! integer indicating maximum detail level to
        ! profile
        INTEGER, parameter :: init_timing_disable_depth = 0 ! init
        INTEGER, private :: timing_disable_depth = init_timing_disable_depth
        ! integer indicating depth of t_disablef calls
        INTEGER, parameter :: init_timing_detail = 0 ! init
        INTEGER, private :: cur_timing_detail = init_timing_detail
        ! current timing detail level
        ! default
        ! flag indicating whether the performance timer
        ! output should be written to a single file
        ! (per component communicator) or to a
        ! separate file for each process
        ! default
        ! maximum number of processes writing out
        ! timing data (for this component communicator)
        ! default
        ! separation between process ids for processes
        ! that are writing out timing data
        ! (for this component communicator)
        ! default
        ! collect and print out global performance statistics
        ! (for this component communicator)
        ! default
        ! integer indicating which timer to use
        ! (as defined in gptl.inc)
        ! default
        ! flag indicating whether the PAPI namelist
        ! should be read and HW performance counters
        ! used in profiling
        ! PAPI counter ids
        ! default
        ! default
        ! default
        ! default
        !=======================================================================
            PUBLIC read_externs_perf_mod
        CONTAINS

        ! module extern variables

        SUBROUTINE read_externs_perf_mod(kgen_unit)
        integer, intent(in) :: kgen_unit
        READ(UNIT=kgen_unit) timing_initialized
        READ(UNIT=kgen_unit) timing_detail_limit
        READ(UNIT=kgen_unit) timing_disable_depth
        READ(UNIT=kgen_unit) cur_timing_detail
        END SUBROUTINE read_externs_perf_mod


        ! read subroutines
        !=======================================================================
        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        SUBROUTINE t_startf(event, handle)
            !-----------------------------------------------------------------------
            ! Purpose: Start an event timer
            ! Author: P. Worley
            !-----------------------------------------------------------------------
            !---------------------------Input arguments-----------------------------
            !
            ! performance timer event name
            CHARACTER(LEN=*), intent(in) :: event
            !
            !---------------------------Input/Output arguments----------------------
            !
            ! GPTL event handle
            INTEGER(KIND=shr_kind_i8), optional :: handle
            !
            !---------------------------Local workspace-----------------------------
            !
            INTEGER :: ierr ! GPTL error return
            !
            !-----------------------------------------------------------------------
            !
            IF ((timing_initialized) .and.        (timing_disable_depth .eq. 0) .and.        (cur_timing_detail .le. &
            timing_detail_limit)) THEN
                IF (present (handle)) THEN
                    !kgen_excluded ierr = gptlstart_handle(event, handle)
                    ELSE
                    !kgen_excluded ierr = gptlstart(event)
                END IF 
            END IF 
            RETURN
        END SUBROUTINE t_startf
        !
        !========================================================================
        !

        SUBROUTINE t_stopf(event, handle)
            !-----------------------------------------------------------------------
            ! Purpose: Stop an event timer
            ! Author: P. Worley
            !-----------------------------------------------------------------------
            !---------------------------Input arguments-----------------------------
            !
            ! performance timer event name
            CHARACTER(LEN=*), intent(in) :: event
            !
            !---------------------------Input/Output arguments----------------------
            !
            ! GPTL event handle
            INTEGER(KIND=shr_kind_i8), optional :: handle
            !
            !---------------------------Local workspace-----------------------------
            !
            INTEGER :: ierr ! GPTL error return
            !
            !-----------------------------------------------------------------------
            !
            IF ((timing_initialized) .and.        (timing_disable_depth .eq. 0) .and.        (cur_timing_detail .le. &
            timing_detail_limit)) THEN
                IF (present (handle)) THEN
                    !kgen_excluded ierr = gptlstop_handle(event, handle)
                    ELSE
                    !kgen_excluded ierr = gptlstop(event)
                END IF 
            END IF 
            RETURN
        END SUBROUTINE t_stopf
        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !
        !========================================================================
        !

        !===============================================================================
    END MODULE perf_mod
