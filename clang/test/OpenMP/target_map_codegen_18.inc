// expected-no-diagnostics
#ifndef HEADER_INC
#define HEADER_INC

// This file is regex-heavy and takes a long time to execute the test. To speed
// testing up, test execution is split over multiple fimes. The RUN commands are
// in the corresponding .cpp files now. Do not add them here.

// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK19

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE00n:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE00n:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE00n:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK19-USE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]
// CK19-NOUSE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-USE: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK19-NOUSE: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-USE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK19-USE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 33]
// CK19-NOUSE: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE08:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK19-USE: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK19-NOUSE: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE10:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-USE: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE11:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-USE: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE12:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i64] [i64 33]
// CK19-NOUSE: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] [i64 32]
// CK19-NOUSE: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 33]
// CK19-NOUSE: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE15:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK19-NOUSE: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE16:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 33]
// CK19-NOUSE: [[MTYPE16:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE17:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 240]
// CK19-USE: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 34]
// CK19-NOUSE: [[SIZE17:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-NOUSE: [[MTYPE17:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE18:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 240]
// CK19-USE: [[MTYPE18:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[SIZE18:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-NOUSE: [[MTYPE18:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE19:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 32]
// CK19-NOUSE: [[MTYPE19:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE20:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 4]
// CK19-USE: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 33]
// CK19-NOUSE: [[SIZE20:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-NOUSE: [[MTYPE20:@.+]] = private {{.*}}constant [1 x i64] [i64 1]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE21:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[MTYPE21:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE22:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 4]
// CK19-USE: [[MTYPE22:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[SIZE22:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-NOUSE: [[MTYPE22:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE23:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE23:@.+]] = private {{.*}}constant [1 x i64] [i64 39]
// CK19-NOUSE: [[MTYPE23:@.+]] = private {{.*}}constant [1 x i64] [i64 7]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE24:@.+]] = private {{.*}}constant [1 x i64] [i64 480]
// CK19-USE: [[MTYPE24:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE24:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE25:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK19-USE: [[MTYPE25:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE25:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE26:@.+]] = private {{.*}}constant [1 x i64] [i64 24]
// CK19-USE: [[MTYPE26:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE26:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE27:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-USE: [[MTYPE27:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE27:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE28:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 16]
// CK19-USE: [[MTYPE28:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]
// CK19-NOUSE: [[MTYPE28:@.+]] = private {{.*}}constant [3 x i64] [i64 3, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE29:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 4]
// CK19-USE: [[MTYPE29:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]
// CK19-NOUSE: [[MTYPE29:@.+]] = private {{.*}}constant [3 x i64] [i64 3, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE30:@.+]] = private {{.*}}constant [4 x i64] [i64 800, i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE30:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE31:@.+]] = private {{.*}}constant [4 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 {{8|4}}, i64 40]
// CK19-USE: [[MTYPE31:@.+]] = private {{.*}}constant [4 x i64] [i64 800, i64 800, i64 800, i64 35]
// CK19-NOUSE: [[SIZE31:@.+]] = private {{.*}}constant [1 x i64] [i64 40]
// CK19-NOUSE: [[MTYPE31:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE32:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19-USE: [[MTYPE32:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE32:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE33:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19-USE: [[MTYPE33:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE33:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE34:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19-USE: [[MTYPE34:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE34:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE35:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE35:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE36:@.+]] = private {{.*}}constant [1 x i64] [i64 208]
// CK19-USE: [[MTYPE36:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE36:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE37:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE37:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE38:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE38:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE39:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE39:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE40:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE40:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE41:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 208]
// CK19-USE: [[MTYPE41:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[SIZE41:@.+]] = private {{.*}}constant [1 x i64] [i64 208]
// CK19-NOUSE: [[MTYPE41:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE42:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 104]
// CK19-USE: [[MTYPE42:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]
// CK19-NOUSE: [[MTYPE42:@.+]] = private {{.*}}constant [3 x i64] [i64 3, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE43:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK19-NOUSE: [[MTYPE43:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE44:@.+]] = private {{.*}}constant [1 x i64] [i64 320]
// CK19-USE: [[MTYPE44:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK19-NOUSE: [[MTYPE44:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK19-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Region 00
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL00:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL00:@.+]]()
  #pragma omp target map(alloc:a)
  {
#ifdef USE
    ++a;
#endif
  }

  // Map of a scalar in nested region.
  int b = a;

  // Region 00n
  // CK19-DAG: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00n]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00n]]{{.+}}, i8** null, i8** null, i32 1, i32 0)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL00n:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL00n:@.+]]()
  #pragma omp target map(alloc:b)
  #pragma omp parallel
  {
#ifdef USE
    ++b;
#endif
  }

  // Map of an array.
  int arra[100];

  // Region 01
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x i32]**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store [100 x i32]* [[VAR0]], [100 x i32]** [[CP0]]

  // CK19-USE: call void [[CALL01:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL01:@.+]]()
  #pragma omp target map(to:arra)
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 02
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 20

  // CK19-USE: call void [[CALL02:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL02:@.+]]()
  #pragma omp target map(from:arra[20:60])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 03
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL03:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL03:@.+]]()
  #pragma omp target map(tofrom:arra[:60])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 04
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL04:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL04:@.+]]()
  #pragma omp target map(alloc:arra[:])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 05
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 15

  // CK19-USE: call void [[CALL05:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL05:@.+]]()
  #pragma omp target map(to:arra[15])
  {
#ifdef USE
    arra[15]++;
#endif
  }

  // Region 06
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i.+ %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19-USE: call void [[CALL06:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL06:@.+]]()
  #pragma omp target map(tofrom:arra[ii:ii+23])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 07
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE07]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i.+ %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL07:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL07:@.+]]()
  #pragma omp target map(alloc:arra[:ii])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 08
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE08]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19-USE: call void [[CALL08:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL08:@.+]]()
  #pragma omp target map(tofrom:arra[ii])
  {
#ifdef USE
    arra[15]++;
#endif
  }

  // Map of a pointer.
  int *pa;

  // Region 09
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK19-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK19-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK19-USE: call void [[CALL09:@.+]](i32** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL09:@.+]]()
  #pragma omp target map(from:pa)
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 10
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 20
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL10:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL10:@.+]]()
  #pragma omp target map(tofrom:pa[20:60])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 11
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE11]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL11:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL11:@.+]]()
  #pragma omp target map(alloc:pa[:60])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 12
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE12]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 15
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL12:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL12:@.+]]()
  #pragma omp target map(to:pa[15])
  {
#ifdef USE
    pa[15]++;
#endif
  }

  // Region 13
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL13:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL13:@.+]]()
  #pragma omp target map(alloc:pa[ii-23:ii])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 14
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL14:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL14:@.+]]()
  #pragma omp target map(to:pa[:ii])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 15
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE15]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE15]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL15:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL15:@.+]]()
  #pragma omp target map(from:pa[ii+12])
  {
#ifdef USE
    pa[15]++;
#endif
  }

  // Map of a variable-size array.
  int va[ii];

  // Region 16
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE16]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z:64|32]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[VAR1]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[VAR0]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL16:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL16:@.+]]()
  #pragma omp target map(to:va)
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 17
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE17]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE17]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 20

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 20

  // CK19-USE: call void [[CALL17:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL17:@.+]]()
  #pragma omp target map(from:va[20:60])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 18
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE18]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE18]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 0

  // CK19-USE: call void [[CALL18:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL18:@.+]]()
  #pragma omp target map(tofrom:va[:60])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 19
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE19]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 0

  // CK19-USE: call void [[CALL19:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL19:@.+]]()
  #pragma omp target map(alloc:va[:])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 20
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE20]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE20]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 15

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 15

  // CK19-USE: call void [[CALL20:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL20:@.+]]()
  #pragma omp target map(to:va[15])
  {
#ifdef USE
   va[15]++;
#endif
  }

  // Region 21
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE21]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} %{{.+}}

  // CK19-USE: call void [[CALL21:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL21:@.+]]()
  #pragma omp target map(tofrom:va[ii:ii+23])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 22
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE22]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE22]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} %{{.+}}

  // CK19-USE: call void [[CALL22:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL22:@.+]]()
  #pragma omp target map(tofrom:va[ii])
  {
#ifdef USE
   va[15]++;
#endif
  }

  // Always.
  // Region 23
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE23]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE23]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL23:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL23:@.+]]()
  #pragma omp target map(always, tofrom: a)
  {
#ifdef USE
   a++;
#endif
  }

  // Multidimensional arrays.
  int marr[4][5][6];
  int ***mptr;

  // Region 24
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE24]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE24]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0]], [4 x [5 x [6 x i32]]]** [[CP0]]

  // CK19-USE: call void [[CALL24:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL24:@.+]]()
  #pragma omp target map(tofrom: marr)
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 25
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE25]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE25]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL25:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL25:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][2:4])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 26
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE26]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE26]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL26:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL26:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][:])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 27
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE27]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE27]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL27:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL27:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][3])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 28
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE28]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE28]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32****
  // CK19-DAG: store i32*** [[VAR0:%.+]], i32**** [[CBP0]]
  // CK19-DAG: store i32*** [[SEC0:%.+]], i32**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32***
  // CK19-DAG: store i32*** [[SEC0]], i32**** [[CBP1]]
  // CK19-DAG: store i32** [[SEC1:%.+]], i32*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i32***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
  // CK19-DAG: store i32** [[SEC1]], i32*** [[CBP2]]
  // CK19-DAG: store i32* [[SEC2:%.+]], i32** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19-USE: call void [[CALL28:@.+]](i32*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL28:@.+]]()
  #pragma omp target map(tofrom: mptr[1][2][2:4])
  {
#ifdef USE
    mptr[1][2][3]++;
#endif
  }

  // Region 29
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE29]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE29]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32****
  // CK19-DAG: store i32*** [[VAR0:%.+]], i32**** [[CBP0]]
  // CK19-DAG: store i32*** [[SEC0:%.+]], i32**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32***
  // CK19-DAG: store i32*** [[SEC0]], i32**** [[CBP1]]
  // CK19-DAG: store i32** [[SEC1:%.+]], i32*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i32***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
  // CK19-DAG: store i32** [[SEC1]], i32*** [[CBP2]]
  // CK19-DAG: store i32* [[SEC2:%.+]], i32** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 3
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19-USE: call void [[CALL29:@.+]](i32*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL29:@.+]]()
  #pragma omp target map(tofrom: mptr[1][2][3])
  {
#ifdef USE
    mptr[1][2][3]++;
#endif
  }

  // Multidimensional VLA.
  double mva[23][ii][ii+5];

  // Region 30
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|4}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[MTYPE30]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  // CK19-64-USE-DAG: [[VAR1]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-USE-DAG: [[VAR11]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i[[Z]]*
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR2:%.+]], i[[Z]]* [[CBP2]]
  // CK19-USE-DAG: store i[[Z]] [[VAR22:%.+]], i[[Z]]* [[CP2]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S2]]
  // CK19-64-USE-DAG: [[VAR2]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-USE-DAG: [[VAR22]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-USE-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to double**
  // CK19-USE-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to double**
  // CK19-USE-DAG: store double* [[VAR3:%.+]], double** [[CBP3]]
  // CK19-USE-DAG: store double* [[VAR3]], double** [[CP3]]
  // CK19-USE-DAG: store i64 [[CSVAL3:%[^,]+]], i64* [[S3]]
  // CK19-USE-DAG: [[CSVAL3]] = {{mul nuw i64 %[^,]+, 8|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double**
  // CK19-NOUSE-DAG: store double* [[VAR0:%.+]], double** [[CBP0]]
  // CK19-NOUSE-DAG: store double* [[VAR0]], double** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 8|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL30:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL30:@.+]]()
  #pragma omp target map(tofrom: mva)
  {
#ifdef USE
    mva[1][2][3]++;
#endif
  }

  // Region 31
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|4}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[SIZE31]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[MTYPE31]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CP0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i[[Z]]*
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR2:%.+]], i[[Z]]* [[CBP2]]
  // CK19-USE-DAG: store i[[Z]] [[VAR22:%.+]], i[[Z]]* [[CP2]]
  //
  // CK19-USE-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to double**
  // CK19-USE-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to double**
  // CK19-USE-DAG: store double* [[VAR3:%.+]], double** [[CBP3]]
  // CK19-USE-DAG: store double* [[SEC3:%.+]], double** [[CP3]]
  // CK19-USE-DAG: [[SEC3]] = getelementptr {{.*}}double* [[SEC33:%.+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC33]] = getelementptr {{.*}}double* [[SEC333:%.+]], i[[Z]] [[IDX3:%.+]]
  // CK19-USE-DAG: [[IDX3]] = mul nsw i[[Z]] %{{[^,]+}}, %{{[^,]+}}
  // CK19-USE-DAG: [[SEC333]] = getelementptr {{.*}}double* [[VAR3]], i[[Z]] [[IDX33:%.+]]
  // CK19-USE-DAG: [[IDX33]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double**
  // CK19-NOUSE-DAG: store double* [[VAR0:%.+]], double** [[CBP0]]
  // CK19-NOUSE-DAG: store double* [[SEC0:%.+]], double** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}double* [[SEC00:%.+]], i[[Z:64|32]] 0
  // CK19-NOUSE-DAG: [[SEC00]] = getelementptr {{.*}}double* [[SEC000:%.+]], i[[Z]] [[IDX0:%.+]]
  // CK19-NOUSE-DAG: [[IDX0]] = mul nsw i[[Z]] %{{[^,]+}}, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[SEC000]] = getelementptr {{.*}}double* [[VAR0]], i[[Z]] [[IDX00:%.+]]
  // CK19-NOUSE-DAG: [[IDX00]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-USE: call void [[CALL31:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL31:@.+]]()
  #pragma omp target map(tofrom: mva[1][ii-2][:5])
  {
#ifdef USE
    mva[1][2][3]++;
#endif
  }

  // Multidimensional array sections.
  double marras[11][12][13];
  double mvlaas[11][ii][13];
  double ***mptras;

  // Region 32
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE32]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE32]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [11 x [12 x [13 x double]]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0]], [11 x [12 x [13 x double]]]** [[CP0]]

  // CK19-USE: call void [[CALL32:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL32:@.+]]()
  #pragma omp target map(marras)
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 33
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE33]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE33]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [12 x [13 x double]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [12 x [13 x double]]* [[SEC0:%.+]], [12 x [13 x double]]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19-USE: call void [[CALL33:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL33:@.+]]()
  #pragma omp target map(marras[:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 34
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE34]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE34]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [12 x [13 x double]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [12 x [13 x double]]* [[SEC0:%.+]], [12 x [13 x double]]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19-USE: call void [[CALL34:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL34:@.+]]()
  #pragma omp target map(marras[:][:][:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 35
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE35]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL35:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL35:@.+]]()
  #pragma omp target map(marras[1][:ii][:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 36
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE36]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE36]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC000]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL36:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL36:@.+]]()
  #pragma omp target map(marras[:1][:2][:13])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 37
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE37]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[VAR2]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL37:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL37:@.+]]()
  #pragma omp target map(mvlaas)
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 38
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE38]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-USE-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC00:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC00]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL38:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL38:@.+]]()
  #pragma omp target map(mvlaas[:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 39
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE39]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-USE-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC00:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC00]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL39:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL39:@.+]]()
  #pragma omp target map(mvlaas[:][:][:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 40
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE40]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-USE-DAG: [[SEC222]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i[[Z]] 0
  // CK19-NOUSE-DAG: [[SEC00]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC000:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC000]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-USE: call void [[CALL40:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL40:@.+]]()
  #pragma omp target map(mvlaas[1][:ii][:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 41
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[SIZE41]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE41]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-USE-DAG: [[SEC222]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2

  // CK19-NO-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NO-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NO-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NO-USE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NO-USE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NO-USE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i[[Z]] 0
  // CK19-NO-USE-DAG: [[SEC00]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC000:%[^,]+]]
  // CK19-NO-USE-DAG: [[SEC000]] = mul nsw i[[Z]] 0, %{{[^,]+}}

  // CK19-USE: call void [[CALL41:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL41:@.+]]()
  #pragma omp target map(mvlaas[:1][:2][:13])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 42
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE42]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE42]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double****
  // CK19-DAG: store double*** [[VAR0:%.+]], double**** [[CBP0]]
  // CK19-DAG: store double*** [[SEC0:%.+]], double**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load double***, double**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}double*** [[SEC00:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
  // CK19-DAG: store double*** [[SEC0]], double**** [[CBP1]]
  // CK19-DAG: store double** [[SEC1:%.+]], double*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}double** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load double**, double*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}double*** [[SEC1111:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC1111]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK19-DAG: store double** [[SEC1]], double*** [[CBP2]]
  // CK19-DAG: store double* [[SEC2:%.+]], double** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}double* [[SEC22:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC22]] = load double*, double** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}double** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load double**, double*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}double*** [[SEC222222:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC222222]] = load double***, double**** [[PTR]],

  // CK19-USE: call void [[CALL42:@.+]](double*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL42:@.+]]()
  #pragma omp target map(mptras[:1][2][:13])
  {
#ifdef USE
    mptras[1][2][3]++;
#endif
  }

  // Region 43 - the memory is not contiguous for this map - will map the whole last dimension.
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE43]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL43:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL43:@.+]]()
  #pragma omp target map(marras[1][:ii][1:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 44
  // CK19-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE44]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE44]]{{.+}}, i8** null, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 20

  // CK19-USE: call void [[CALL44:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL44:@.+]]()
  #pragma omp target map(from:arra[20:])
  {
#ifdef USE
    arra[50]++;
#endif
  }

}

// CK19: define {{.+}}[[CALL00]]
// CK19: define {{.+}}[[CALL01]]
// CK19: define {{.+}}[[CALL02]]
// CK19: define {{.+}}[[CALL03]]
// CK19: define {{.+}}[[CALL04]]
// CK19: define {{.+}}[[CALL05]]
// CK19: define {{.+}}[[CALL06]]
// CK19: define {{.+}}[[CALL07]]
// CK19: define {{.+}}[[CALL08]]
// CK19: define {{.+}}[[CALL09]]
// CK19: define {{.+}}[[CALL10]]
// CK19: define {{.+}}[[CALL11]]
// CK19: define {{.+}}[[CALL12]]
// CK19: define {{.+}}[[CALL13]]
// CK19: define {{.+}}[[CALL14]]
// CK19: define {{.+}}[[CALL15]]
// CK19: define {{.+}}[[CALL16]]
// CK19: define {{.+}}[[CALL17]]
// CK19: define {{.+}}[[CALL18]]
// CK19: define {{.+}}[[CALL19]]
// CK19: define {{.+}}[[CALL20]]
// CK19: define {{.+}}[[CALL21]]
// CK19: define {{.+}}[[CALL22]]
// CK19: define {{.+}}[[CALL23]]
// CK19: define {{.+}}[[CALL24]]
// CK19: define {{.+}}[[CALL25]]
// CK19: define {{.+}}[[CALL26]]
// CK19: define {{.+}}[[CALL27]]
// CK19: define {{.+}}[[CALL28]]
// CK19: define {{.+}}[[CALL29]]
// CK19: define {{.+}}[[CALL30]]
// CK19: define {{.+}}[[CALL31]]
// CK19: define {{.+}}[[CALL32]]
// CK19: define {{.+}}[[CALL33]]
// CK19: define {{.+}}[[CALL34]]
// CK19: define {{.+}}[[CALL35]]
// CK19: define {{.+}}[[CALL36]]
// CK19: define {{.+}}[[CALL37]]
// CK19: define {{.+}}[[CALL38]]
// CK19: define {{.+}}[[CALL39]]
// CK19: define {{.+}}[[CALL40]]
// CK19: define {{.+}}[[CALL41]]
// CK19: define {{.+}}[[CALL42]]
// CK19: define {{.+}}[[CALL43]]
// CK19: define {{.+}}[[CALL44]]

#endif // CK19
#endif // HEADER_INC
